import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DroneFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that processes both numerical drone state data and
    visual features from YOLO detections.

    This hybrid architecture:
    1. Processes numerical state features (position, velocity, orientation) with an MLP
    2. Processes YOLO visual features with a CNN (expects image features or detection maps)
    3. Concatenates both feature sets and passes them through a final MLP layer
    """

    def __init__(self, observation_space, cnn_output_dim=64, features_dim=128):
        super(DroneFeatureExtractor, self).__init__(observation_space, features_dim)

        # Determine input dimensions - assuming observation space is a Box
        # and the last 10 values are from YOLO
        self.state_dim = observation_space.shape[0] - 10  # First part is drone state
        self.yolo_dim = 10  # Last 10 values are YOLO detections

        # For numerical state data (position, velocity, orientation)
        self.state_net = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # For YOLO detection data (treating it as a 1D signal for CNN)
        # Reshape YOLO data to [batch_size, 1, yolo_dim] for 1D convolution
        self.yolo_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate output dimensions of the YOLO CNN
        with torch.no_grad():
            sample_input = torch.zeros(1, 1, self.yolo_dim)
            yolo_output_dim = self.yolo_net(sample_input).shape[1]

        # Combine outputs from both networks
        self.combined_net = nn.Sequential(
            nn.Linear(64 + yolo_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Split observations into state and YOLO parts
        state_data = observations[:, :self.state_dim]
        yolo_data = observations[:, self.state_dim:]

        # Process state data
        state_features = self.state_net(state_data)

        # Process YOLO data - reshape for 1D CNN
        batch_size = yolo_data.shape[0]
        yolo_data_reshaped = yolo_data.view(batch_size, 1, self.yolo_dim)
        yolo_features = self.yolo_net(yolo_data_reshaped)

        # Combine features
        combined = torch.cat([state_features, yolo_features], dim=1)
        return self.combined_net(combined)
