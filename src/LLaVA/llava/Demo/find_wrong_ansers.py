import json
questions_json_path =r'D:\Projects\FastV\FastV\src\LLaVA\datas\querys\coco_pope_random.json'
answers_json_path1 = r'D:\Projects\FastV\FastV\src\LLaVA\datas\answers\coco_pope_random_with_answers_0-35.json'
answers_json_path2 = r'D:\Projects\FastV\FastV\src\LLaVA\datas\answers\coco_pope_random_with_answers_35-576.json'
# 加载问题集
with open(questions_json_path, 'r') as file:
    questions = json.load(file)

# 加载答案集1
with open(answers_json_path1, 'r') as file:
    answers1 = {item['question_id']: item['label'] for item in json.load(file)}

# 加载答案集2
with open(answers_json_path2, 'r') as file:
    answers2 = {item['question_id']: item['label'] for item in json.load(file)}
    
# 准备一个列表来存储不一致的问题
inconsistent_questions = []

# 遍历问题集
for question in questions:
    question_id = question['question_id']
    correct_answer = question['label'].lower()
    # 获取两个答案集中的答案
    answer1 = answers1.get(question_id).lower()
    answer2 = answers2.get(question_id).lower()

    # 检查答案是否不一致
    if answer1 and answer2 and answer1 != answer2:
        if answer2 == correct_answer and answer1 != correct_answer:
            inconsistent_questions.append(question)

# 打印结果
print("不一致的问题集：")
for q in inconsistent_questions:
    print(q)