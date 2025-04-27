import torch, torchvision, PIL, io, base64, os
from concurrent.futures import ThreadPoolExecutor
import litserve as ls

precision = torch.bfloat16

class ImageClassifierAPI(ls.LitAPI):
    def setup(self, device):
        weights = torchvision.models.ResNet152_Weights.DEFAULT
        self.image_processing = weights.transforms()
        self.model = torchvision.models.resnet152(weights=weights).eval().to(device).to(precision)
        self.pool = ThreadPoolExecutor(os.cpu_count())

    def decode_request(self, request):
        image_data = request["image_data"]
        return image_data

    def batch(self, inputs):
        print(len(inputs))
        def process_batch(image_data):
            image = base64.b64decode(image_data)
            pil_image = PIL.Image.open(io.BytesIO(image)).convert("RGB")
            return self.image_processing(pil_image)

        batched_inputs = list(self.pool.map(process_batch, inputs))
        return torch.stack(batched_inputs).to(self.device).to(precision)

    def predict(self, x):
        with torch.inference_mode():
            outputs = self.model(x)
            _, predictions = torch.max(outputs, 1)
        return predictions

    def unbatch(self, output):
        return output.tolist()

    def encode_response(self, output):
        return {"output": output}

if __name__ == "__main__":
    api = ImageClassifierAPI()
    server = ls.LitServer(api, accelerator="gpu", devices=1, max_batch_size=16, workers_per_device=8, batch_timeout=0.01)
    server.run(port=8000, num_api_servers=4)
