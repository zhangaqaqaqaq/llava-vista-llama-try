import json

# 定义文件路径

ans_file = r'D:\Projects\FastV\FastV\src\LLaVA\datas\answers\coco_pope_random_with_answers_35-576.json'
label_file = r'D:\Projects\LLaVA\datas\querys\coco_pope_random.json'
# 读取answers_ad.json文件
with open(ans_file, 'r') as f:
    answers = json.load(f)

# 读取coco_pope_adversarial.json文件
with open(label_file, 'r') as f:
    label_list = json.load(f)

# 创建一个字典，以question_id为键，将标签和答案存储起来
answers_dict = {answer['question_id']: {'label': answer['label'].lower()} for answer in answers}
labels_dict = {label['question_id']: {'label': label['label'].lower()} for label in label_list}

# 初始化统计变量
TP = TN = FP = FN = 0

# 遍历答案字典，比较label和answer是否匹配
for question_id in answers_dict:
    if question_id in labels_dict:
        answer_label = answers_dict[question_id]['label']
        label = labels_dict[question_id]['label']
        answer_label = answer_label.lower()
        if 'yes' in answer_label:
            answer_label = "yes"
        if 'no' in answer_label:
            answer_label = "no"
        # 检查答案是否匹配
        if answer_label == label:
            if answer_label == 'yes':
                TP += 1
            else:
                TN += 1
        else:
            if answer_label == 'yes':
                FP += 1
            else:
                FN += 1
    else:
        print(f"Warning: question_id {question_id} from answers not found in labels.")

# 计算性能指标
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
yes = (TP + FP) / (TP + TN + FP + FN)
# 打印结果
print('TP\tFP\tTN\tFN\t')
print(f'{TP}\t{FP}\t{TN}\t{FN}')
print(f'Accuracy: {acc:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 score: {f1:.4f}')
print(f'yes%: {yes:.4f}')