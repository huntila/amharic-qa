# Modify Haystack squad v2 for computing P, R, and F1 scores using eval.py script.
# If it is not normalized the eval.py script raises 'Missing answer <question_id>'
import json


def haystack_squadv2_to_squadv2(haystack_squadv2_file, output_file):
    haystack_squadv2_json = json.load(open(haystack_squadv2_file, encoding='utf-8'))
    data = []
    for haystack_squadv2_example in haystack_squadv2_json['data']:
        for p in haystack_squadv2_example['paragraphs']:
            context = p['context']
            temp = []
            for q in p['qas']:
                question_text = q['question']
                para = {'qas': [{'question': question_text, 'answers': []}]}
                qa = para['qas'][0]
                qa['id'] = str(q['id'])
                qa['is_impossible'] = True

                if "answers" in q:
                    ans_string = q['answers'][0]['text']
                    start_char_index = q['answers'][0]['answer_start']
                    qa['answers'].append({'text': ans_string, 'answer_start': start_char_index})
                    qa['is_impossible'] = False
                temp.append(para['qas'][0])
            para2 = {'context': context, 'qas': temp}
            data.append({'paragraphs': [para2]})

    haystack_squadv2_as_squadv2 = {'data': data, 'version': '2.0'}
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(haystack_squadv2_as_squadv2, outfile, indent=2, sort_keys=True, ensure_ascii=False)
    print('Successfully modified!')


haystack_squadv2_to_squadv2('test_data.json', 'modified_test_data.json')