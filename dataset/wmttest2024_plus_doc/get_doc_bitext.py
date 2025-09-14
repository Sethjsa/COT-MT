import xml.etree.ElementTree as ET
import shutil
import os


def create_clean_dir(path):
    """
    Create a clean directory. If the directory exists, remove it first.
    :param path: Path of the directory to create.
    """
    # Remove the directory if it exists
    if os.path.exists(path):
        shutil.rmtree(path)
    # Create the directory
    os.makedirs(path)


lang_pair = "en-zh"

lang_pairs = ["en-cs", "en-de", "en-fr", "en-he", "en-ja", "en-ru", "en-uk", "en-zh"]

for lang_pair in lang_pairs:

    # output
    output_dir = os.path.join("", f"{lang_pair}")
    create_clean_dir(output_dir)

    # wmt24 sentence data
    wmt24_src_data = "../wmttest2024_plus/gpt-4o-mini/step_by_step/{}/src".format(lang_pair)
    wmt24_tgt_data = "../wmttest2024_plus/gpt-4o-mini/step_by_step/{}/tgt".format(lang_pair)
    wmt24 = {}
    with open(wmt24_src_data) as src_fin, open(wmt24_tgt_data) as tgt_fin:
        for src, tgt in zip(src_fin, tgt_fin):
            src = src.strip()
            tgt = tgt.strip()
            wmt24[src] = tgt

    # wmt24 meta data, use en-zh for all
    tree = ET.parse('raw/wmttest2024.src.{}.xml'.format("en-zh"))
    root = tree.getroot()

    sentences = []
    cnt = 0

    doc_cnt = 0
    literary_cnt = 0
    news_cnt = 0
    social_cnt = 0
    speech_cnt = 0
    s, t = lang_pair.split('-')
    with open(os.path.join(output_dir, "test.{}.{}".format(lang_pair, s)), "w") as src_fout, open(os.path.join(output_dir, "test.{}.{}".format(lang_pair, t)), "w") as tgt_fout:

        for doc in root.findall(".//doc"):
            doc_id = doc.get('id')
            # if "-literary_" in doc_id:
            #     literary_cnt += 1
            # if "-speech_" in doc_id:
            #     speech_cnt += 1
            # if "-news_" in doc_id:
            #     news_cnt += 1
            # if "-social_" in doc_id:
            #     social_cnt += 1

            sub_doc_src = []
            sub_doc_tgt = []
            cnt = 0
            for seg in doc.findall(".//seg"):
                seg_id = seg.get('id')
                text = seg.text.strip()
                if text in wmt24:
                    if cnt + len(text.split()) < 150:
                        cnt += len(text.split())
                        sub_doc_src.append(text)
                        sub_doc_tgt.append(wmt24[text])
                    else:
                        print(" ".join(sub_doc_src))
                        print(" ".join(sub_doc_tgt))
                        src_fout.write(" ".join(sub_doc_src) + "\n")
                        tgt_fout.write(" ".join(sub_doc_tgt) + "\n")
                        sub_doc_src = [text]
                        sub_doc_tgt = [wmt24[text]]
                        cnt = len(text.split())
                        doc_cnt += 1
            if len(sub_doc_src) != 0:
                print(" ".join(sub_doc_src))
                print(" ".join(sub_doc_tgt))
                src_fout.write(" ".join(sub_doc_src) + "\n")
                tgt_fout.write(" ".join(sub_doc_tgt) + "\n")
                doc_cnt += 1
