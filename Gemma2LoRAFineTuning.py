import keras
import keras_nlp
import os
import io
import pandas as pd

os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

# from google.colab import files
# uploaded = files.upload()

file_path = '/content/example_5000-6000.csv'
# 尝试使用不同的编码格式读取文件
try:
#     df = pd.read_csv(file_path, encoding='utf-8')  # 默认编码
# except UnicodeDecodeError:
#     df = pd.read_csv(file_path, encoding='ISO-8859-1')  # 西欧语言
# except UnicodeDecodeError:
#     df = pd.read_csv(file_path, encoding='latin1')  # 西欧语言
# except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='cp1252')  # 西欧和北美
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='GBK')  # 简体中文
print(df.head(10))  # 打印前10行

column_names = ['ask', 'answer']
# 使用loc来读取指定的列
ask_answer = df.loc[:, column_names]

data = []
for index, row in ask_answer.iterrows():
    # 这里可以访问每行的数据，row是一个Series对象
    # Format the entire example as a single string.
    template = "Instruction:\n{ask}\n\nResponse:\n{answer}"
    data.append(template.format(**row))

# Only use 1000 training examples, to keep it fast.
data = data[:1000]
print(data)

os.environ['KAGGLE_USERNAME'] = ''  # 替换为你的Kaggle用户名
os.environ['KAGGLE_KEY'] = ''       # 替换为你的Kaggle API密钥

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma2_2b_en")
print(gemma_lm.summary())

# Enable LoRA for the model and set the LoRA rank to 4.
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()

# Limit the input sequence length to 256 (to control memory usage).
gemma_lm.preprocessor.sequence_length = 256
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.fit(data, epochs=1, batch_size=1)

# Uncomment the line below if you want to enable mixed precision training on GPUs
# keras.mixed_precision.set_global_policy('mixed_bfloat16')

prompt = template.format(
    ask="治疗癫痫医院？现在每天都很着就难过我的孩子以后会跟疾病交朋也了解过许多医生的救可是孩子身上的癫痫疾病还是复发了好几次小孩癫痫怎么治疗好？",
    answer="",
)
sampler = keras_nlp.samplers.TopKSampler(k=5, seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=256))

