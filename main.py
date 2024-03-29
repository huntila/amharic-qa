from transformers import XLMRobertaTokenizer, XLMRobertaForQuestionAnswering, pipeline
import torch
import json
from pathlib import Path
import gradio as gr

tokenizer = XLMRobertaTokenizer.from_pretrained('aware-ai/xlmroberta-squadv2')
model = XLMRobertaForQuestionAnswering.from_pretrained('aware-ai/xlmroberta-squadv2')

question = "በላሊበላ የጌታ ልደት ቀን በቤተ ማርያም የሚቀርበው ልዩ ዝማሬ ምን ይባላል?"
context = "ንጉሡ ላሊበላ የሚለውን ስም ያገኘው፣ ሲወለድ በንቦች ስለተከበበ ነው። ላል ማለት ማር ማለት ሲሆን፤ ላሊበላ ማለትም -ላል ይበላል (ማር ይበላል) ማለት አንደሆነ " \
          "ይነግራል።  ውቅር ቤተክርስቲያናቱን ንጉሡ ጠርቦ የስራቸው ከመላእክት እገዛ ጋር እንደሆነ በኢትዮጵያ ኦርቶዶክስ እምነት ተከታዮች ይነግራል። በ16ኛው ከፍለ ዘመን " \
          "አውሮፓዊ ተጓዥ ላሊበላን ተመልክቶ «ያየሁትን ብናግር ማንም እንደኔ ካላየ በፍጹም አያምነኝም» ሲል ተናግሮ ነበር። በላሊበላ 11 ውቅር ዐብያተ ክርስቲያናት ያሉ ሲሆን " \
          "ከነዚህም ውስጥ ቤተ ጊዮርጊስ (ባለ መስቀል ቅርፁ) ሲታይ ውሃልኩን የጠበቀ ይመስላል። ቤተ መድሃኔ ዓለም የተባለው ደግሞ ከሁሉም ትልቁ ነው። ላሊበላ (ዳግማዊ " \
          "ኢየሩሳሌም) የገና በዓል ታህሳስ 29 በልዩ ሁኔታ ና ድምቀት ይከበራል፣ \"ቤዛ ኩሉ\" ተብሎ የሚጠራው በነግህ የሚደረገው ዝማሬ በዚሁ በዓል የሚታይ ልዩ ና ታላቅ " \
          "ትዕይንት ነው።የሚደረገውም ከቅዳሴ በኋላ በቤተ ማርያም ሲሆን ከታች ባለ ነጭ ካባ ካህናት ከላይ ደግሞ ባለጥቁር ካብ ካህናት በቅዱስ ያሬድ ዜማ ቤዛ ኩሉ እያሉ " \
          "ይዘምራሉ። 11ዱ የቅዱስ ላሊበላ ፍልፍል አብያተ ክርስቲያናት ቤተ መድሃኔ ዓለም፣ ቤተ ማርያም፣ ቤተ ደናግል፣ ቤተ መስቀል፣ ቤተ ደብረሲና፣ ቤተ ጎለጎታ፣ ቤተ " \
          "አማኑኤል፣ ቤተ አባ ሊባኖስ፣ ቤተ መርቆሬዎስ፣ ቤተ ገብርኤል ወሩፋኤል፣ ቤተ ጊዮርጊስ ናቸው። "


def answer_extraction(context, question):
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


context2 = "አፈወርቅ ተክሌ የዓለም ሊቀ ሊቃውንት፣ የኪነ ጥበብ ምሑር የተከበሩ ሰዓሊ አፈወርቅ ተክሌ በዓለም የታወቁና የተከበሩት ኢትዮጵያዊ  ሰዓሊ አፈወርቅ ተክሌ በሰሜን ሸዋ " \
           "በምትገኘው የሸዋ ነገሥታት ከተማ አንኮበር ላይ ጥቅምት ፲፫ ቀን ፲፱፻፳፭ ዓ/ም ከአባታቸው አቶ ተክሌ  ማሞ እና ከእናታቸው ከወይዘሮ ፈለቀች የማታወርቅ ተወለዱ። ገና " \
           "በጨቅላ ዕድሜያቸው ፋሺስት ኢጣልያ ኢትዮጵያን በግፍ ሲወራት፣ የልጅነት ትዝታቸውም ይሄው የግፍ ወረራ እና ጦርነት የሚያስከትለው የሰው፣ የንብረት እና የባህል ጥፋት ፤ " \
           "ከነጻነትም በኋላ ዐቢዩ ሥራ በወረራ የተበላሸችውን ሀገራቸውን እንደገና መገንባት እንደሆነ ነው። ለዚህም የሀገር ግንባታ አስፈላጊውን እውቀት መሸመት የፈለጉት በ " \
           "ማዕድን ምሕንድስና ዘርፍ ነበር። ወላጆቻቸውና ዘመድ አዝማዶች ግን የኪነ ጥበብ ስጦታቸውን በቤታቸውና በከተማው ዙሪያ በሚስሏቸው ስዕሎች ተገንዝበውት ነበር። ገና " \
           "በአሥራ አምስት ዓመታቸው በ፲፱፻፵ ዓ/ም ለከፍተኛ ትምህርት ተመርጠው የምሕንድስና ትምህርታቸውን ለመከታተል ወደ እንግሊዝ አገር ይላካሉ። አፈወርቅ ተክሌ በእንግሊዝ " \
           "አገር የአዳሪ ተማሪ ቤትን ኑሮ ሲያስታውሱ የባዕድ ባህላት፣ የአየር ለውጥ እና የተለመደው የተማሪ ቤት ዝንጠላ እንዳስቸገራቸው ያወሳሉ። ቢሆንም ትምሕርታቸውን በትጋት " \
           "ሲከታተሉ ቆዩ። በተለይም በሒሣብ፣ በኬሚስትሪ እና በታሪክ ትምሕርቶች ጥሩ ውጤት በማምጣት ሰለጠኑ። ነገር ግን አስተማሪዎቻቸው የተፈጥሮ ስጦታቸው ኪነ ጥበብ እንደሆነ " \
           "ለመገንዘብ ብዙ ጊዜም አልወሰደባቸውም። በነሱም አበረታችነት በእዚህ ስጦታቸው ላይ የበለጠ ለማተኮር ወስነው በሎንዶን የኪነ ጥበብ ማእከላዊ ትምሕርት ቤት ተመዝግበው " \
           "ገቡ እዚህ ትምህርት ቤት ጥምሕርታቸውን ሲያጠናቅቁ ለከፍተኛ ጥናት በስመ ጥሩው የሎንዶን ዩኒቨርሲቲ የስሌድ ኪነ ጥበብ ማዕከል የመጀመሪያው አፍሪቃዊ ተማሪ (" \
           "ከተከተሏቸው አፍሪቃውያን አንዱ የሱዳን ተወላጁ  ኢብራሂም ኤል ሳላሂ ናቸው) በመሆን ገቡ። እዚህ በስዕል፣ ቅርጽ እና በአርክቴክቸር ጥናቶች ላይ አተኩረው ተመረቁ። " \
           "ትምሕርታቸውን አጠናቀው ወደ ኢትዮጵያ ሲመለሱ፣ በየጥቅላይ ግዛቶቹ እየተዘዋወሩ፣ በየቦታው እስከ ሦስት ወራት በመቀመጥ የኢትዮጵያን ታሪክ እና የብሔረሰቦቿን ባህልና " \
           "ወግ በሚገባ አጥንተዋል። "
question2 = "አፈወርቅ ተክሌ ፲፱፻፵ ዓ.ም ለከፍተኛ ት/ት ተመርጠው ወደ እንግሊዝ ሀገር የተላኩት ምን እንዲያጠኑ ነበር?"
with gr.Blocks() as demo:
    gr.Interface(
        answer_extraction,
        title="Amharic Question Answering Demo",
        inputs=[
            gr.Textbox(lines=7, value=context, label="ጽሑፍ (Context)"),
            gr.Textbox(lines=2, value=question, label="ጥያቄ (Question)"),
        ],
        outputs=[gr.Textbox(label="መልስ (Answer)")],
        examples=[
            [context, question],
            [context2, question2]
        ],
    )


# , gr.Textbox(label="Score")

def main():
    # test_data_file = 'test_data.json'
    # print(amqa.answer_extraction(question, context))
    # answers = run_answer_extractor(test_data_file)
    # # Write the prediction result of the model to a file
    # with open('predictions.json', 'w', encoding='utf-8') as outfile:
    #     json.dump(answers, outfile, indent=2, sort_keys=False, ensure_ascii=False)
    # print(answers)
    # # Modify the test_data format appropriate for the evaluation
    # utils.haystack_squadv2_to_squadv2('test_data.json', 'modified_test_data.json')
    demo.launch(share=True)


if __name__ == '__main__':
    main()
