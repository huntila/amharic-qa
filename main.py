from transformers import XLMRobertaTokenizer, XLMRobertaForQuestionAnswering
import torch
import json
from pathlib import Path
import utils

tokenizer = XLMRobertaTokenizer.from_pretrained('aware-ai/xlmroberta-squadv2')
model = XLMRobertaForQuestionAnswering.from_pretrained('aware-ai/xlmroberta-squadv2')

question, context = "ግርማዊት ዘውዲቱ ንግሥተ ነገሥት ዘኢትዮጵያ ሆነው ሲነግሱ እንደራሴ የተደረጉት ንጉሥ የተወለዱት የት ነው?", \
                    "ቀዳማዊ ዓፄ ኃይለ ሥላሴ (ቀ.ኃ.ሥ.) ከጥቅምት ፳፫ ቀን ፲፱፻፳፫ እስከ መስከረም ፪ ቀን ፲፱፻፷፯ ዓ.ም. የኢትዮጵያ ንጉሠ ነገሥት ነበሩ። ተፈሪ መኰንን " \
                    "ሐምሌ ፲፮ ቀን ፲፰፻፹፬ ዓ.ም. ከአባታቸው ከልዑል ራስ መኰንን እና ከእናታቸው ከወይዘሮ የሺ እመቤት ኤጀርሳ ጎሮ በተባለ የገጠር ቀበሌ ሐረርጌ ውስጥ " \
                    "ተወለዱ።\nበ፲፰፻፺፱  የሲዳሞ  አውራጃ  አገረ  ገዥ  ሆኑ። በ፲፱፻፫ ዓ.ም.  የሐረርጌ  አገረ  ገዥ  ሆኑ። የሐረርጌ  አገረ  ገዥ  ሲሆኑ ፣ በጣም  " \
                    "ብዙ  ሺህ  ተከታዮች  ነበሩዋቸውና  ልጅ  እያሱን  ከእንደራሴነት  በኅይል  እንዳያስወጡ ፤ እያሱም  ተፈሪን  ከሐረር  አገረ  ገዥነት  እንዳይሽሯቸው  " \
                    "የሚል  ስምምነት  ተዋዋሉ። ዳሩ  ግን  እያሱ  ሃይማኖታቸውን  ከክርስትና  ወደ  እስልምና  እንደቀየሩ  የሚል  ማስረጃ  ቀረበና  ብዙ  መኳንንትና  " \
                    "ቀሳውስት  ስለዚህ  ኢያሱን  አልወደዱዋቸውም  ነበር። ከዚህም  በላይ  እያሱ  ተፈሪን  ከሐረር  ከአገረ  ገዥነታቸው  ለመሻር  በሞከሩበት  ወቅት  " \
                    "ስምምነታቸው  እንግዲህ  ተሠርዞ  ተፈሪ  ደግሞ  ለወገናቸው  ከስምምነቱ  ተለቅቀው  በዚያን  ጊዜ  እሳቸው  እያሱን  ከእንደራሴነት  አስወጡ። እንግዲህ  " \
                    "በ፲፱፻፱  ዓ.ም.  መኳንንቱ  ዘውዲቱን  ንግሥተ  ነገሥት  ሆነው  አድርገዋቸው  ተፈሪ  ደግሞ  እንደራሴ  ሆኑ። ከዚህ  ወቅት  ጀምሮ  ተፈሪ  " \
                    "በኢትዮጵያ  ውስጥ  ባለሙሉ  ሥልጣን  ነበሩ። በመስከረም ፳፯  ቀን  ፲፱፻፳፩  ዓ.ም.  የንጉሥነት  ማዕረግ  ተጨመረላችው። በ፲፱፻፳፪ ዓ.ም  ንግሥት  " \
                    "ዘውዲቱ  አርፈው  ንጉሠ  ነገሥት  ሆኑና  ጥቅምት ፳፫  ቀን  ፲፱፻፳፫ ዓ.ም.  ብዙ  የውጭ  ልዑካን  በተገኙበት  ታላቅ  ሥነ-ሥርዓት  ቅብዓ  ቅዱስ  " \
                    "ተቀብተው  እሳቸውና  ሚስታቸው  እቴጌ  መነን  ዘውድ  ጫኑ። \nበንግሥ  በዓሉ  ዋዜማ  ጥቅምት ፳፪  ቀን  የትልቁ  ንጉሠ-ነገሥት  የዳግማዊ  ዓፄ  " \
                    "ምኒልክ  ሐውልት  በመናገሻ  ቅዱስ  ጊዮርጊስ  ቤተ-ክርስቲያን  አጠገብ ፤ ለዘውድ  በዓል  የመጡት  እንግዶች  በተገኙበት  ሥርዐት ፣ የሐውልቱን  " \
                    "መጋረጃ  የመግለጥ  ክብር ለብሪታንያ  ንጉሥ  ወኪል  ለ(ዱክ  ኦፍ  ግሎስተር)  ተሰጥቶ  ሐውልቱ  ተመረቀ።\nለንግሥ  ስርዐቱ  ጥሪ  የተደረገላቸው  " \
                    "የውጭ  አገር  ልኡካን  ከየአገራቸው  ጋዜጠኞች  ጋር  ከጥቅምት ፰  ቀን  ጀምሮ  በየተራ  ወደ  አዲስ  አበባ  ገብተው  ስለነበር  ሥርዓቱ  በዓለም  " \
                    "ዜና  ማሰራጫ  በየአገሩ  ታይቶ  ነበር። በተለይም  በብሪታንያ  ቅኝ  ግዛት  በጃማይካ  አንዳንድ  ድሀ  ጥቁር  ሕዝቦች  ስለ  ማዕረጋቸው  ተረድተው  " \
                    "የተመለሰ  መሢህ  ነው  ብለው  ይሰብኩ  ጀመር። እንደዚህ  የሚሉት  ሰዎች  እስከ  ዛሬ  ድረስ  ስለ  ፊተኛው  ስማቸው  «ራስ  ተፈሪ»  ትዝታ  " \
                    "ራሳቸውን  «ራስታፋራይ»  (ራሰተፈሪያውያን)  ብለዋል።\nንግሥተ ነገሥታት ዘውዲቱ\nግርማዊት ንግሥተ ነገሥታት ዘውዲቱ ቅዳሜ ሚያዝያ ፳፪ ቀን ፲፰፻፷፰ " \
                    "ዓ.ም ከሸዋ ንጉሥ ምኒልክና ከወረኢሉ ተወላጅ ከወይዘሮ አብቺው በተጉለትና ቡልጋ አውራጃ በሞረትና ጅሩ ወረዳ እነዋሪ ከተማ አጠገብ በምትገኝ ሰገነት በምትባል " \
                    "መንደር ተወለዱ። ሐምሌ ፲፩ ቀን ሰኞ ሥርዓተ ጥምቀት ተፈጽሞላቸው፣ የክርስትና ስማቸው አስካለ ማርያም ተባለ። ግርማዊት ንግሥተ ነገሥት ዘውዲቱ ለዳግማዊ " \
                    "ምኒልክ ሦስተኛ ልጅ ሲሆኑ ከልጅ ኢያሱ ቀጥለው መኳንንትና ሕዝቡ መክሮና ዘክሮ በምርጫ በኢትዮጵያ ዙፋን ላይ የተቀመጡ ብቸኛዋ ንግሥት ናቸው።\nአባታቸው " \
                    "የሸዋው ንጉሥ ምኒልክ (በኋላ ዳግማዊ ዐፄ) ዘውዲቱ የስድስት ዓመት ሕጻን እንደነበሩ ለአሥራ ሁለት ዓመቱ የዐፄ ዮሐንስ ልጅ ለአርአያ ሥላሴ ዳሩዋቸው። " \
                    "ጋብቻቸውም ጥቅምት ፲፫ ቀን ፲፰፻፸፭ ዓ.ም. በተክሊልና በቁርባን ተፈጸመ የሠርጉም ማማርና የሽልማቱ ብዛት ሊነገር አይቻልም። ልጅ ኢያሱ መስከረም ፲፯ ቀን " \
                    "፲፱፻፱ ዓ.ም ከስልጣን ሲወርዱ፤ መኳንንቱና ሕዝቡ መክሮና ዘክሮ በምርጫ ወይዘሮ ዘውዲቱን ግርማዊት ንግሥተ ነገሥት፤ ራስ ተፈሪ መኮንንን አልጋ ወራሽ እና " \
                    "ባለሙሉ ሥልጣን እንደራሴ እንዲሆኑ ወስነው መስከረም ፳፩ ቀን \"ግርማዊት ዘውዲቱ፣ ንግሥተ ነገሥት ዘኢትዮጵያ\" ተብለው ነገሡ። በመኳንንቱም ምክር መሠረት " \
                    "ጳጳሱና እጨጌ ወልደጊዮርጊስ \"ኢያሱን የተከተልክ፣ ያነገሥናትን ዘውዲቱንና ራስ ተፈሪን የከዳህ፣ ውግዝ ከመአርዮስ\" እያሉ በአዋጅ አወገዙ[1]\nከዚህ " \
                    "በኋላ በጥቅምት የሰገሌ ጦርነት ከተደረገ በኋላ የዘውድ በዓሉ የካቲት ፬ ቀን ፲፱፻፱ ዓ.ም እንዲሆን ታዞ፣ ለዘውዱ በዓል በኢትዮጵያ አዋሳኝ ያሉት የእንግሊዝ " \
                    "ሱዳንና የእንግሊዝ ሱማሌ ገዥዎች፤ የፈረንሳይ ሱማሌ ገዥ እና በኢትዮጵያም ታላላቅ ራሶች እንዲሁም የየአውራጃው ገዥዎች በተገኙበት በዕለተ እሑድ አባታቸው " \
                    "ባሠሩት በትልቁ ደብር በመናገሻ ቅዱስ ጊዮርጊስ ቤተ ክርስቲያን በሊቀ ጳጳሱ በአቡነ ማቴዎስ እጅ ቅብዐ መንግስት ተቀብተው የንጉሠ ነገሥቱን ዘውድ ጫኑ። " \
                    "ከዚህም በኋላ እንደ ሥርዓተ ንግሡ ሕግ፤ ንግሥተ ነገሥት ዘውዲቱ በአባታቸው ዙፋን ተቀምጠው መንገሣቸውን፤ ደጃዝማች ተፈሪም ራስ ተብለው አልጋ ወራሽና ባለሙሉ " \
                    "ሥልጣን እንደራሴ መሆናቸውን በቤተ ክርስቲያኑ ቅጥር ግቢ በተሰበሰበው ሕዝብ ፊት አዋጅ ተነገረ። \nየንግሥተ ነገሥታትን ሥልጣን ለማሳወቅ ሲባል ራስ ወልደ " \
                    "ጊዮርጊስ የካቲት ፲፩ ቀን ራስ ወልደ ጊዮርጊስን ንጉሠ ጎንደር ብለው አንግሠው ዘውድ ደፉላቸው። እንደዚሁም መስከረም ፳፯ ቀን 1921 ዓ.ም. ብዙ መኳንንትና " \
                    "መሣፍንት፤ ሊቃውንትና ጳጳሳት ባሉበት የኢትዮጵያ መንግሥት አልጋ ወራሽ ራስ ተፈሪ መኮንንን አዲስ አበባ መካነ ሥላሴ ቤተ ክርስቲያን፣ ንጉሥ ተፈሪ ተብለው " \
                    "ዘውድ ተጫነላቸው። "


def answer_extraction(question, context):
    encoding = tokenizer(question, context, return_tensors='pt', truncation=True,
                         padding=True)  # truncation=True is added for resolving the size (743>512) problem
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    start_scores, end_scores = model(input_ids, attention_mask=attention_mask, output_attentions=False)[:2]

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])
    answer = tokenizer.convert_tokens_to_ids(answer.split())
    answer = tokenizer.decode(answer)
    # print(question,answer)
    return answer


def run_answer_extractor(test_data_file_name):
    path = Path(test_data_file_name)
    with open(path, 'rb') as f:
        data_dict = json.load(f)
    prediction = {}
    for group in data_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                question_id = qa['id']
                model_answer = answer_extraction(question, context)
                prediction.update({question_id: model_answer})
    return prediction


def main():
    test_data_file = 'test_data.json'
    # print(answer_extraction(question, context))
    answers = run_answer_extractor(test_data_file)
    # Write the prediction result of the model to a file
    with open('predictions.json', 'w', encoding='utf-8') as outfile:
        json.dump(answers, outfile, indent=2, sort_keys=False, ensure_ascii=False)
    print(answers)
    # Modify the test_data format appropriate for the evaluation
    utils.haystack_squadv2_to_squadv2('test_data.json', 'modified_test_data.json')


if __name__ == '__main__':
    main()
