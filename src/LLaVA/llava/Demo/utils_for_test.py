import torch
import math

def safe_multiply_matrix(A, B):
    # 检查输入是否为 PyTorch 张量
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise ValueError("Both inputs must be PyTorch tensors")
    
    # 保存原始数据类型
    original_dtype = A.dtype if A.dtype == B.dtype else None
    
    # 确保输入张量是浮点数类型
    A = A.float()
    B = B.float()
    
    # 执行矩阵乘法
    result = torch.matmul(A, B)
    
    # 将 -inf * 0 的结果替换为 0
    # 这里我们使用逻辑索引来找到所有 -inf 值，并将其替换为 0
    result[result == float('-inf')] = 0
    
    # 如果原始数据类型不是浮点数，将结果转换回原始数据类型
    if original_dtype is not None and original_dtype != torch.float32:
        result = result.to(original_dtype)
    
    return result

# 创建两个示例张量
A = torch.randint(0, 10, (2, 3, 4), dtype=torch.int32)
B = torch.randint(0, 10, (2, 4, 5), dtype=torch.int32)

# 测试函数
C = safe_multiply_matrix(A, B)
print(C)