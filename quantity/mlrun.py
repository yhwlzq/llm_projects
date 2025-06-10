import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from mlflow.pyfunc import PythonModel

class QwenAWQWrapper(PythonModel):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def predict(self, context, model_input):
        if isinstance(model_input, str):
            input_text = [model_input]
        elif isinstance(model_input, list):
            input_text = model_input
        else:
            input_text = model_input["text"].tolist() if hasattr(model_input, "text") else [str(model_input)]
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# 1. Configure device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load AWQ quantized model
model_path = "/home/mystic/PycharmProjects/finetuning/ragbi/quantity/qwen2-7b-awq"

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    exit(1)

# 3. Log to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = "Qwen2_7B_AWQ_1"

try:
    # Get or create experiment (handles deleted state)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    elif experiment.lifecycle_stage == "deleted":
        # Handle deleted experiment by creating a new one with same name
        print(f"Experiment '{experiment_name}' was deleted, creating new one")
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    with mlflow.start_run(experiment_id=experiment_id):
        # Log parameters
        mlflow.log_params({
            "model_type": "qwen2_7b",
            "quant_method": "awq",
            "device": device,
            "framework": "transformers+awq"
        })
        
        # Log the model
        mlflow.pyfunc.log_model(
            artifact_path="qwen_awq",
            python_model=QwenAWQWrapper(model, tokenizer),
            registered_model_name="qwen2_7b_awq",
            input_example={"text": "介绍一下量子计算"},
            pip_requirements=[
                "torch",
                "transformers>=4.34.0",
                "autoawq>=0.1.8",
                "mlflow>=2.8.0"
            ]
        )
        
        print(f"模型记录成功！Run ID: {mlflow.active_run().info.run_id}")
        print(f"UI 地址: {mlflow.get_tracking_uri()}")

except Exception as e:
    print(f"记录失败: {e}")
    # You might want to check MLflow server status here
    print("请确保MLflow服务器正在运行，并且地址正确")
finally:
    if mlflow.active_run():
        mlflow.end_run()