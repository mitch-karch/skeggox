import cv2

# from turbojpeg import TurboJPEG
from base_camera import BaseCamera
import torch, timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


class SingleInstance:
    def __init__(self, arch, device, model_file_path):
        self.arch = arch
        self.device = device
        self.model_file_path = model_file_path
        self.create_model()
        self.load_model()
        self.transform = self.get_transforms()

    def load_model(self):
        filename = self.model_file_path  #'checkpoints/model_best.pth.tar'
        model = self.model
        if os.path.isfile(filename):
            print(f"=> loading checkpoint: {filename}")
            checkpoint = torch.load(filename, map_location=self.device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
        else:
            print(f"=> no checkpoint found at {filename}")
            raise Exception(f"=> no checkpoint found at {filename} | {os.getcwd()}")

    def create_model(self):
        model = timm.create_model(self.arch, pretrained=False)
        self.model = model.to(self.device)

    def predict(self, image):
        img = self.transform_image(image).to(self.device, dtype=torch.float)
        with torch.no_grad():
            return self.model(img.unsqueeze(0))

    def transform_image(self, img):
        image = self.transform(image=img)["image"]
        image = image.float() / 255.0
        return image

    @staticmethod
    def get_transforms():
        return A.Compose(
            [
                A.Resize(224, 224),
                A.Blur(always_apply=True, p=1),
                ToTensorV2(),
            ]
        )


class Camera(BaseCamera):
    def __init__(self):
        super(Camera, self).__init__()

    @staticmethod
    def frames():
        # jpeg = TurboJPEG()
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 16, 240)
        thickness = 2
        text_position = (120, 130)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_file_path = "checkpoints/checkpoint.pth.tar"
        single_instance = SingleInstance(
            arch="mixnet_xl", device=device, model_file_path=model_file_path
        )

        camera = cv2.VideoCapture(
            'udpsrc port=9000 caps="application/x-rtp, media=(string)video, payload=(int)96, clock-rate=(int)90000, encoding-name=(string)H264"'
            "! rtph264depay"
            "! video/x-h264,width=1024,height=768,framerate=25/1"
            "! h264parse"
            "! avdec_h264"
            # "! nvv4l2decoder"
            # "! nvvidconv"
            "! videoconvert"
            # "! video/x-raw, format=(string)I420"
            "! appsink ",
            cv2.CAP_GSTREAMER,
        )
        dummy_counter = 0
        last_pred =  -1
        while True:
            dummy_counter += 1
            ret, img = camera.read()
            # img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_I420)

            if ret:
                if dummy_counter % 10 == 0:
                    predicts = single_instance.predict(img)
                    last_pred = torch.cat([predicts]).argmax(1).detach().cpu().numpy()[0]
                    if last_pred == 10:
                        last_pred = "Empty"
                    elif last_pred == 11:
                        last_pred = "Clutch"
                img = cv2.putText(
                    img, str(last_pred), text_position, font, 2, color, thickness, cv2.LINE_AA
                )
                yield cv2.imencode(".jpg", img)[1].tobytes()
            # yield jpeg.encode(img)
