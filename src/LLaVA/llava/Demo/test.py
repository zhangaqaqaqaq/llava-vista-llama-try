import torch


torch.manual_seed(0)



attention_a = torch.rand(1, 2, 4, 4)
attention_b = torch.rand(1, 2, 4, 4)

v_mask = torch.tensor([1, 1, 0, 0])


v_mask = v_mask.unsqueeze(0).unsqueeze(1).unsqueeze(-1)

part1 = attention_a * v_mask
part2 = attention_b * (1 - v_mask)
new_attention = v_mask * attention_a + (1 - v_mask) * attention_a

# 打印结果
print("part1:\n", str(part1))
print("part2:\n", str(part2))
print("attention_a:\n", str(attention_a))
print("attention_b:\n", str(attention_b))
print("v_mask:\n", str(v_mask))
print("new_attention:\n", str(new_attention))