from torch import nn
from torchscope import scope
from torchvision import models


class FaceAttributeModel(nn.Module):
    def __init__(self):
        super(FaceAttributeModel, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(model.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(1280, 17)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, images):
        x = self.resnet(images)  # [N, 1280, 1, 1]
        x = self.pool(x)
        x = x.view(-1, 1280)  # [N, 1280]
        x = self.fc(x)
        reg = self.sigmoid(x[:, :5])  # [N, 8]
        expression = self.softmax(x[:, 5:8])
        gender = self.softmax(x[:, 8:10])
        glasses = self.softmax(x[:, 10:13])
        race = self.softmax(x[:, 13:17])
        return reg, expression, gender, glasses, race


if __name__ == "__main__":
    model = FaceAttributeModel()
    scope(model, input_size=(3, 224, 224))
