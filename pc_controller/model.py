"""
PyTorch ActionNet — Classification model for discrete driving commands.

Since keyboard data only has ~6-8 unique command pairs, this is actually
a CLASSIFICATION problem, not regression. This model predicts which
discrete action to take: forward, backward, left, right, forward-left,
forward-right, backward-left, backward-right, or stop.

Input: 66x200x3 RGB image (cropped)
Output: probabilities for each driving action
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Discrete action definitions ──────────────────────────────
# Each action maps to (left_motor, right_motor) in range [-100, 100]
# These are the common commands from WASD driving
ACTION_TABLE = [
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

NUM_ACTIONS = len(ACTION_TABLE)


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
    for i, (al, ar) in enumerate(ACTION_TABLE):
        dist = (left - al)**2 + (right - ar)**2
        if dist < best_dist:
            best_dist = dist
            best_action = i
    return best_action


def action_to_command(action_idx, speed=70):
    """Convert an action index back to (left, right) motor command at given speed."""
    base_l, base_r = ACTION_TABLE[action_idx]
    # Scale by speed ratio
    scale = speed / 70.0
    return (int(base_l * scale), int(base_r * scale))


class ActionNet(nn.Module):
    """
    Classification CNN for discrete driving actions.
    Same conv backbone as PilotNet but with classification head.
    """

    def __init__(self, num_actions=NUM_ACTIONS):
        super(ActionNet, self).__init__()

        self.num_actions = num_actions

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

        # Classification head — moderate regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(64 * 1 * 18, 64),
            nn.ELU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(64, num_actions),
        )

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
        returns: (batch, num_actions) raw logits
        """
        x = self.features(x)
        x = self.feat_dropout(x)
        x = self.classifier(x)
        return x

    def predict_action(self, x):
        """Get the predicted action index."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_command(self, x, speed=70):
        """Get motor commands directly from image."""
        action = self.predict_action(x).item()
        return action_to_command(action, speed)


# ── Keep backward compat: PilotNet is now ActionNet ──
PilotNet = ActionNet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ActionNet()
    print(f"ActionNet - {count_parameters(model):,} parameters")
    print(f"Actions: {NUM_ACTIONS} classes")
    for i, (l, r) in enumerate(ACTION_TABLE):
        print(f"  [{i}] L={l:+4d} R={r:+4d}")

    dummy = torch.randn(1, 3, 66, 200)
    logits = model(dummy)
    probs = F.softmax(logits, dim=1)
    action = torch.argmax(probs, dim=1).item()
    cmd = action_to_command(action)
    print(f"\nInput:  {dummy.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Action: {action} -> L={cmd[0]}, R={cmd[1]}")
