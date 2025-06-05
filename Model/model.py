import torch
import torch.nn as nn
import constants


class BlinkModelMed(nn.Module):
    def __init__(
        self,
        num_features: int = constants.Model_Constants.NUM_FEATURES,
        img_height: int = constants.Image_Constants.IM_HEIGHT,
        img_width: int = constants.Image_Constants.IM_WIDTH,
        conv_im_channels=constants.Model_Constants.MEDIUM_MODEL_CONSTANTS.CONVOLUTIONAL_IMAGE_CHANELS,
        conv_fc_channels=constants.Model_Constants.MEDIUM_MODEL_CONSTANTS.CONVOLUTIONAL_NUMERIC_CHANELS,
        lstm_hidden: int = constants.Model_Constants.MEDIUM_MODEL_CONSTANTS.LTSM_HIDDEN,
        lstm_layers: int = constants.Model_Constants.MEDIUM_MODEL_CONSTANTS.LTSM_LAYERS,
        bidirectional: bool = constants.Model_Constants.MEDIUM_MODEL_CONSTANTS.BIDIRECTIONAL,
    ):
        super().__init__()

        self.conv_1 = nn.Conv2d(1, conv_im_channels[0], 3, padding=1)
        self.conv_2 = nn.Conv2d(conv_im_channels[0], conv_im_channels[1], 3, padding=1)
        self.conv_3 = nn.Conv2d(conv_im_channels[1], conv_im_channels[2], 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

        conv_out_h = img_height // 4
        conv_out_w = img_width // 4
        flat_dim = conv_out_h * conv_out_w * conv_fc_channels[1]
        self.img_fc = nn.Linear(
            flat_dim, constants.Model_Constants.MEDIUM_MODEL_CONSTANTS.IMAGE_OUTPUT_SIZE
        )

        self.num_fc_1 = nn.Linear(num_features, conv_fc_channels[0])
        self.num_fc_2 = nn.Linear(num_features, conv_fc_channels[1])

        self.lstm = nn.LSTM(
            constants.Model_Constants.MEDIUM_MODEL_CONSTANTS.LTSM_INPUT_SIZE,
            lstm_hidden,
            lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)
        self.bi = bidirectional

    # ————————————————————————————————————————————
    def forward(self, eye_seq, num_seq):
        """Inputs
        eye_seq : (batch,time_steps,1,24,12) – float32
        num_seq : (batch,time_steps,7)
        returns   (batch,) logit
        """
        batch, time_steps, channels, height, width = eye_seq.shape
        img_prossesing = eye_seq.view(batch * time_steps, channels, height, width)
        img_prossesing = self.pool(self.relu(self.conv_1(img_prossesing)))
        img_prossesing = self.pool(self.relu(self.conv_2(img_prossesing)))
        img_prossesing = self.relu(
            self.img_fc(img_prossesing.view(batch * time_steps, -1))
        )

        number_prossesing = self.relu(
            self.num_fc_1(num_seq.view(batch * time_steps, -1))
        )
        number_prossesing = self.relu(
            self.num_fc_2(num_seq.view(batch * time_steps, -1))
        )
        combined_processing = torch.cat(
            [img_prossesing, number_prossesing], dim=1
        ).view(batch, time_steps, -1)

        _, (hidden_states, _) = self.lstm(combined_processing)
        seq_repetition = (
            torch.cat([hidden_states[-2], hidden_states[-1]], dim=1)
            if self.bi
            else hidden_states[-1]
        )
        return self.fc(seq_repetition).squeeze(1)


class BlinkDetectorSmall(nn.Module):
    def __init__(
        self,
        num_features: int = constants.Model_Constants.NUM_FEATURES,
        img_height: int = constants.Image_Constants.IM_HEIGHT,
        img_width: int = constants.Image_Constants.IM_WIDTH,
        conv_im_channels=constants.Model_Constants.SMALL_MODEL_CONSTANTS.CONVOLUTIONAL_IMAGE_CHANELS,
        lstm_hidden: int = constants.Model_Constants.SMALL_MODEL_CONSTANTS.LTSM_HIDDEN,
        lstm_layers: int = constants.Model_Constants.SMALL_MODEL_CONSTANTS.LTSM_LAYERS,
        bidirectional: bool = constants.Model_Constants.SMALL_MODEL_CONSTANTS.BIDIRECTIONAL,
    ):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, conv_im_channels[0], 3, padding=1)
        self.conv_2 = nn.Conv2d(conv_im_channels[0], conv_im_channels[1], 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

        conv_out_h = img_height // 4  # 24 -> 6
        conv_out_w = img_width // 4  # 12 -> 3
        flat_dim = conv_out_h * conv_out_w * conv_im_channels[1]

        self.img_fc = nn.Linear(
            flat_dim, constants.Model_Constants.SMALL_MODEL_CONSTANTS.IMAGE_OUTPUT_SIZE
        )

        self.num_fc_1 = nn.Linear(
            num_features,
            constants.Model_Constants.SMALL_MODEL_CONSTANTS.CONVOLUTIONAL_NUMERIC_CHANELS[
                -1
            ],
        )
        self.lstm = nn.LSTM(
            constants.Model_Constants.SMALL_MODEL_CONSTANTS.LTSM_INPUT_SIZE,
            lstm_hidden,
            lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

        self.bi = bidirectional

    def forward(self, eye_seq, num_seq):
        batch, time_steps, channels, height, width = eye_seq.shape
        img_prossesing = eye_seq.view(batch * time_steps, channels, height, width)
        img_prossesing = self.pool(self.relu(self.conv_1(img_prossesing)))
        img_prossesing = self.pool(self.relu(self.conv_2(img_prossesing)))
        img_prossesing = self.relu(
            self.img_fc(img_prossesing.view(batch * time_steps, -1))
        )

        number_prossesing = self.relu(
            self.num_fc_1(num_seq.view(batch * time_steps, -1))
        )
        combined_processing = torch.cat(
            [img_prossesing, number_prossesing], dim=1
        ).view(batch, time_steps, -1)

        _, (hidden_states, _) = self.lstm(combined_processing)
        seq_repetition = (
            torch.cat([hidden_states[-2], hidden_states[-1]], dim=1)
            if self.bi
            else hidden_states[-1]
        )
        return self.fc(seq_repetition).squeeze(1)


# only uses number data (smallest model)
class BlinkDetectorXS(nn.Module):
    def __init__(
        self,
        num_features: int = constants.Model_Constants.NUM_FEATURES,
        img_height: int = constants.Image_Constants.IM_HEIGHT,
        img_width: int = constants.Image_Constants.IM_WIDTH,
        conv_im_channels=constants.Model_Constants.XS_MODEL_CONSTANTS.CONVOLUTIONAL_IMAGE_CHANELS,
        lstm_hidden: int = constants.Model_Constants.XS_MODEL_CONSTANTS.LTSM_HIDDEN,
        lstm_layers: int = constants.Model_Constants.XS_MODEL_CONSTANTS.LTSM_LAYERS,
        bidirectional: bool = constants.Model_Constants.XS_MODEL_CONSTANTS.BIDIRECTIONAL,
    ):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, conv_im_channels[0], 3, padding=1)
        self.conv_2 = nn.Conv2d(conv_im_channels[0], conv_im_channels[1], 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

        conv_out_h = img_height // 4  # 24 -> 6
        conv_out_w = img_width // 4  # 12 -> 3
        flat_dim = conv_out_h * conv_out_w * conv_im_channels[1]

        self.num_fc_1 = nn.Linear(num_features, 64)
        self.num_fc_2 = nn.Linear(num_features, 32)
        self.num_fc_3 = nn.Linear(num_features, 16)

        self.lstm = nn.LSTM(
            constants.Model_Constants.XS_MODEL_CONSTANTS.LTSM_INPUT_SIZE,
            lstm_hidden,
            lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

        self.bidirectional = bidirectional

    def forward(self, eye_seq, num_seq):
        batch, time_steps, channels, height, width = eye_seq.shape

        number_prossesing = self.relu(
            self.num_fc_1(num_seq.view(batch * time_steps, -1))
        )
        number_prossesing = self.relu(
            self.num_fc_2(num_seq.view(batch * time_steps, -1))
        )
        number_prossesing = self.relu(
            self.num_fc_3(num_seq.view(batch * time_steps, -1))
        )

        _, (hidden_states, _) = self.lstm(number_prossesing)
        if self.bidirectional:
            seq_repetition = torch.cat([hidden_states[-2], hidden_states[-1]], dim=1)
        else:
            seq_repetition = hidden_states[-1]
        return self.fc(seq_repetition)
