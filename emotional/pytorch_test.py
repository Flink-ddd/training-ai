import torch
import torch.nn as nn
import os

os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCH_DYNAMO_VERBOSE"] = "1"

class EmptyForwardModel(nn.Module):
    def forward(self, x, *args, **kwargs):
        pass

class EmptyForwardModel_1(nn.Module):
    def forward(self, x):
        pass

model = EmptyForwardModel()
model_1 = EmptyForwardModel_1()

if __name__ == "__main__":
    example_input = (torch.randn(1, 3, 224, 224),)
    # ... (导出尝试)
    try:
        # 这个会失败
        exported_model = torch.export.export(model, example_input)
        print(f"Exported model is exported successfully.")
    except Exception as e:
        print(f"[EXPORT FAILED]: model, {e}")

    try:
        # 这个会成功
        exported_model_1 = torch.export.export(model_1, example_input)
        print(f"Exported model_1 is exported successfully.")
    except Exception as e:
        print(f"[EXPORT FAILED]: model_1 ", e)