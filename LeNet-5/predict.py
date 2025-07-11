import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import matplotlib

# 配置中文字体（仅使用系统必装字体，避免警告）
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]  # 移除Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def load_model(model_path):
    """加载ONNX模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"模型加载成功: {model_path}")
    print(f"输入名称: {input_name}, 输出名称: {output_name}")

    return session, input_name, output_name


def preprocess_image(image, target_size=(32, 32)):
    """图像预处理：支持路径或Image对象"""
    if isinstance(image, str):
        image = Image.open(image).convert('L')
    elif isinstance(image, Image.Image):
        image = image.convert('L')
    else:
        raise TypeError("image必须是文件路径或PIL.Image对象")

    image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    image_tensor = np.expand_dims(np.expand_dims(image_array, axis=0), axis=0)

    return image_resized, image_tensor


def predict_digit(session, input_name, image_tensor):
    """预测数字：使用softmax计算概率"""
    outputs = session.run(None, {input_name: image_tensor})
    logits = outputs[0][0]

    # 计算概率分布
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    predicted_class = np.argmax(probabilities)
    confidence = np.max(probabilities) * 100

    return predicted_class, confidence, probabilities


def visualize_result(image, predicted_class, confidence, probabilities):
    """可视化结果：确保中文正常显示"""
    plt.figure(figsize=(10, 4))

    # 显示图像和预测结果
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'预测结果: {predicted_class}\n置信度: {confidence:.2f}%')
    plt.axis('off')

    # 显示概率分布
    plt.subplot(1, 2, 2)
    plt.bar(range(10), probabilities, color='skyblue')
    plt.xticks(range(10))
    plt.xlabel('数字类别')
    plt.ylabel('预测概率')
    plt.title('各类别的预测概率分布')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def main():
    model_path = "lenet.onnx"
    test_image_path = "picture/1.jpg"  # 替换为你的图片路径

    session, input_name, output_name = load_model(model_path)
    image_resized, image_tensor = preprocess_image(test_image_path)
    predicted_class, confidence, probabilities = predict_digit(session, input_name, image_tensor)

    # 优化输出格式
    print(f"预测结果: {predicted_class}")
    print(f"置信度: {confidence:.2f}%\n")

    print("各类别概率（百分比）:")
    for digit in range(10):
        print(f"数字 {digit}: {probabilities[digit] * 100:.2f}%")  # 百分比格式

    visualize_result(image_resized, predicted_class, confidence, probabilities)


if __name__ == "__main__":
    main()