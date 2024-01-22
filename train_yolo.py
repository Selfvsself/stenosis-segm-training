from ultralytics import YOLO


def train():
    model = YOLO("yolov8x-seg.pt")
    results = model.train(data="dataset/data/data.yaml", epochs=200, device='cuda', batch=16)


if __name__ == '__main__':
    train()
