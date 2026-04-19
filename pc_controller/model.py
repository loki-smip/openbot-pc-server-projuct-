"""
PyTorch ActionNet — Dual-head classification model for driving.

Head 1 (Motor):  9 classes — discrete WASD motor combos
Head 2 (Servo):  9 classes — discrete tilt positions (0°→180°)

Both heads share the same convolutional backbone (PilotNet-style).

Input:  66×200×3 RGB image (cropped)
Output: (motor_logits [9], servo_logits [9])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SERVO_POSITIONS, NUM_SERVO_POSITIONS


# ── Discrete motor action definitions ────────────────────────
# Each action maps to (left_motor, right_motor) in range [-100, 100]
# These are the common commands from WASD driving
MOTOR_ACTION_TABLE = [
    (  0,   0),   # 0: STOP
    ( 70,  70),   # 1: FORWARD
    (-70, -70),   # 2: BACKWARD
    (-49,  49),   # 3: TURN LEFT (pivot)
    ( 49, -49),   # 4: TURN RIGHT (pivot)
    ( 21,  70),   # 5: FORWARD + LEFT (arc)
    ( 70,  21),   # 6: FORWARD + RIGHT (arc)
    (-21, -70),   # 7: BACKWARD + LEFT
    (-70, -21),   # 8: BACKWARD + RIGHT
]

NUM_MOTOR_ACTIONS = len(MOTOR_ACTION_TABLE)

# ── Backward compatibility aliases ───────────────────────────
ACTION_TABLE = MOTOR_ACTION_TABLE
NUM_ACTIONS = NUM_MOTOR_ACTIONS


def servo_to_class(servo_angle):
    """
    Map a continuous servo angle (0-180) to the nearest discrete class.
    Returns an index into SERVO_POSITIONS.
    """
    best_idx = 0
    best_dist = abs(servo_angle - SERVO_POSITIONS[0])
    for i, pos in enumerate(SERVO_POSITIONS):
        dist = abs(servo_angle - pos)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def class_to_servo(class_idx):
    """Convert a servo class index back to an angle."""
    if 0 <= class_idx < len(SERVO_POSITIONS):
        return SERVO_POSITIONS[class_idx]
    return 90  # fallback to center


def command_to_action(left, right, speed=70):
    """
    Map a (left, right) motor command to the nearest discrete action index.
    Works with any speed value — normalizes by comparing the pattern.
    """
    if abs(left) < 5 and abs(right) < 5:
        return 0  # STOP

    # Normalize to pattern
    max_val = max(abs(left), abs(right), 1)
    nl = left / max_val   # -1 to 1
    nr = right / max_val  # -1 to 1

    # Match patterns
    if nl > 0.3 and nr > 0.3:
        if abs(nl - nr) < 0.3:
            return 1  # FORWARD
        elif nl < nr:
            return 5  # FORWARD + LEFT
        else:
            return 6  # FORWARD + RIGHT
    elif nl < -0.3 and nr < -0.3:
        if abs(nl - nr) < 0.3:
            return 2  # BACKWARD
        elif abs(nl) < abs(nr):
            return 7  # BACKWARD + LEFT
        else:
            return 8  # BACKWARD + RIGHT
    elif nl < -0.2 and nr > 0.2:
        return 3  # TURN LEFT (pivot)
    elif nl > 0.2 and nr < -0.2:
        return 4  # TURN RIGHT (pivot)

    # Fallback: find nearest by distance
    best_action = 0
    best_dist = float('inf')
    for i, (al, ar) in enumerate(MOTOR_ACTION_TABLE):
        dist = (left - al)**2 + (right - ar)**2
        if dist < best_dist:
            best_dist = dist
            best_action = i
    return best_action


def action_to_command(action_idx, speed=70):
    """Convert a motor action index back to (left, right) command at given speed."""
    base_l, base_r = MOTOR_ACTION_TABLE[action_idx]
    # Scale by speed ratio
    scale = speed / 70.0
    return (int(base_l * scale), int(base_r * scale))


class ActionNet(nn.Module):
    """
    Dual-head classification CNN for driving.

    Shared convolutional backbone (PilotNet-style) with two classifier heads:
      - Motor head: predicts one of 9 motor actions
      - Servo head: predicts one of 9 tilt positions
    """

    def __init__(self, num_motor_actions=NUM_MOTOR_ACTIONS,
                 num_servo_positions=NUM_SERVO_POSITIONS):
        super(ActionNet, self).__init__()

        self.num_motor_actions = num_motor_actions
        self.num_servo_positions = num_servo_positions

        # Keep backward compat attribute
        self.num_actions = num_motor_actions

        # Convolutional feature extractor (same as PilotNet)
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ELU(inplace=True),

            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ELU(inplace=True),

            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ELU(inplace=True),

            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
        )

        self.feat_dropout = nn.Dropout2d(0.15)

        # Shared flatten + first dense layer
        self._flat_size = 64 * 1 * 18  # from 66x200 input

        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(self._flat_size, 128),
            nn.ELU(inplace=True),
            nn.Dropout(0.35),
        )

        # Motor classification head
        self.motor_head = nn.Linear(128, num_motor_actions)

        # Servo classification head
        self.servo_head = nn.Linear(128, num_servo_positions)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (batch, 3, 66, 200) float tensor in [0, 1]
        returns: (motor_logits, servo_logits)
            motor_logits: (batch, num_motor_actions)
            servo_logits: (batch, num_servo_positions)
        """
        x = self.features(x)
        x = self.feat_dropout(x)
        x = self.shared_fc(x)
        motor_logits = self.motor_head(x)
        servo_logits = self.servo_head(x)
        return motor_logits, servo_logits

    def predict(self, x):
        """Get predicted motor action index and servo class index."""
        with torch.no_grad():
            motor_logits, servo_logits = self.forward(x)
            motor_action = torch.argmax(motor_logits, dim=1)
            servo_class = torch.argmax(servo_logits, dim=1)
            return motor_action, servo_class

    def predict_command(self, x, speed=70):
        """Get (left, right, servo_angle) directly from image."""
        motor_action, servo_class = self.predict(x)
        left, right = action_to_command(motor_action.item(), speed)
        servo_angle = class_to_servo(servo_class.item())
        return left, right, servo_angle


# ── Keep backward compat: PilotNet is now ActionNet ──
PilotNet = ActionNet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ActionNet()
    print(f"ActionNet (Dual-Head) - {count_parameters(model):,} parameters")
    print(f"Motor actions: {NUM_MOTOR_ACTIONS} classes")
    for i, (l, r) in enumerate(MOTOR_ACTION_TABLE):
        print(f"  [{i}] L={l:+4d} R={r:+4d}")
    print(f"Servo positions: {NUM_SERVO_POSITIONS} classes")
    for i, pos in enumerate(SERVO_POSITIONS):
        print(f"  [{i}] {pos}°")

    dummy = torch.randn(1, 3, 66, 200)
    motor_logits, servo_logits = model(dummy)
    motor_probs = F.softmax(motor_logits, dim=1)
    servo_probs = F.softmax(servo_logits, dim=1)
    motor_action = torch.argmax(motor_probs, dim=1).item()
    servo_class = torch.argmax(servo_probs, dim=1).item()
    left, right = action_to_command(motor_action)
    servo_angle = class_to_servo(servo_class)
    print(f"\nInput:  {dummy.shape}")
    print(f"Motor logits: {motor_logits.shape}")
    print(f"Servo logits: {servo_logits.shape}")
    print(f"Motor action: {motor_action} -> L={left}, R={right}")
    print(f"Servo class:  {servo_class} -> {servo_angle}°")
