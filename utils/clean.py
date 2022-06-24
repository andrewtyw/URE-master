
import sys
import os
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(CURR_DIR)
P = PATH.parent
for i in range(3): # add parent path, height = 3
    P = P.parent
    sys.path.append(str(P.absolute()))



import copy
import re
import random
import string
from tqdm import tqdm 
from distutils.log import error

def find_uppercase(str1:list, str2:list):
	"""
		str1 is a longer text
		str2(only lower case) is text included in str1
		find text in str1 that consist with str2, ignoring case. Then return the text in str1
	"""
	l2 = len(str2)
	for i in range(len(str1[:-l2])+1):
		if ' '.join(str1[i:i+l2]).lower().split()==str2:
			return ' '.join(str1[i:i+l2])
	return ' '.join(str2)



def wiki_check_punctuation(text:str,punc=(string.punctuation+'–¡„“')):

    
    
    text_new = ""
    for _,symbol in enumerate(text):
        if symbol in punc:
            text_new+=(" "+symbol+" ")
        else:
            text_new+=symbol
    text_new = ' '.join(text_new.split())
    text_new = text_new.replace('" -','"-').replace('. . .','...')
    return text_new


def replace_s(text,ori,target):
    ori = wiki_check_punctuation(ori)
    insensitive = re.compile(re.escape(ori), re.IGNORECASE)
    res= insensitive.sub(target, text,1)
    return res

def replace_s_tac(text,ori,target):
    insensitive = re.compile(re.escape(ori), re.IGNORECASE)
    res= insensitive.sub(target, text,1)
    return res

def replace_s_1(text,ori,target):
    
    
    index = -1
    try:
        text_sp = text.lower().split()
        ori_sp = ori.lower().split()
        L_ori = len(ori_sp)
        index = -1
        for i in range(len(text_sp)):
            if text_sp[i:i+L_ori]==ori_sp:
                index = i
                break
        res = text.split()[:index]+target.split()+text.split()[index+L_ori:]
        res = ' '.join(res)
        if index==-1:
            
            
            
            
            
            return re.sub(ori, target, text, flags=re.IGNORECASE) 
    except:
        
        
        
        

        
        res = re.sub(ori, target, text, flags=re.IGNORECASE)
    return res





def eliminate_noRelation(data, rel_key='rel'):
    not_norelation_index = []
    for i, rel in enumerate(data[rel_key]):
        if rel != 'no_relation':
            not_norelation_index.append(i)
    for k, _ in data.items():
        data[k] = [data[k][val_idx] for val_idx in not_norelation_index]
    return data
def text_adjust(text):

    
    
    return text

def assure_replace(text:str):

    if (not text.find("<S:")>=0) and (not text.find("<O:")>=0):
        return -3
    if not text.find("<S:")>=0:
        return -1
    if not text.find("<O:")>=0:
        return -2
    return 1
def obj_prefix(obj, obj_type): return " <O:{}> {} </O:{}> ".format(
    obj_type, obj, obj_type)

def subj_prefix(obj, obj_type): return " <S:{}> {} </S:{}> ".format(
    obj_type, obj, obj_type)

def get_format_train_text_tac(data: dict,text_k="text", subj_k="subj",
                          obj_k="obj", subj_t_k="subj_type", obj_t_k="obj_type",subj_pos = "subj_pos", obj_pos = "obj_pos"):
 
    index = 0
    formatted_texts = []
    all_prefix = set()
    for text, obj, subj, obj_t,subj_t,obj_p,subj_p  in tqdm(zip(data[text_k], data[obj_k], data[subj_k],
                                                            data[obj_t_k], data[subj_t_k],
                                                            data[obj_pos], data[subj_pos]),total=len(data[text_k])):
        origin_text = copy.deepcopy(text)
        
        text = text.split()
        obj_before = text[:obj_p[0]]
        obj_text = text[obj_p[0]:obj_p[1]+1]
        assert obj_text==obj.split()
        obj_text = obj_prefix(" ".join(obj_text),obj_t).split()
        obj_after = text[obj_p[1]+1:]
        text = ' '.join(obj_before+obj_text+obj_after)
        
        if max(subj_p)<=min(obj_p):  
            text = text.split()
            subj_before = text[:subj_p[0]]
            subj_text = text[subj_p[0]:subj_p[1]+1]
            assert subj_text==subj.split()
            subj_text = subj_prefix(" ".join(subj_text),subj_t).split()
            subj_after = text[subj_p[1]+1:]
            text = ' '.join(subj_before+subj_text+subj_after)
        elif max(obj_p)<=min(subj_p): 
            text = text.split()
            subj_before = text[:subj_p[0]+2]
            subj_text = text[subj_p[0]+2:subj_p[1]+2+1]
            assert subj_text==subj.split()
            subj_text = subj_prefix(" ".join(subj_text),subj_t).split()
            subj_after = text[subj_p[1]+2+1:]
            text = ' '.join(subj_before+subj_text+subj_after)
        else: 
            text = replace_s_tac(text,subj,subj_prefix(subj, subj_t))
        assert assure_replace(text)>0
        
        if assure_replace(text)<0:
            print(index,assure_replace(text))
        origin_text_1 = []
        
        for w in text.split():
            w:str
            if not w.startswith(("</O:","<O:","</S:","<S:")):
                origin_text_1.append(w)
        origin_text_1 = " ".join(origin_text_1)
        assert origin_text_1==origin_text
        formatted_texts.append(text)
        all_prefix.add("<O:{}>".format(obj_t))
        all_prefix.add("<S:{}>".format(subj_t))
        all_prefix.add("</O:{}>".format(obj_t))
        all_prefix.add("</S:{}>".format(subj_t))
        index+=1
    return formatted_texts,all_prefix

def get_correctly_dotted_sentence(text:str):
    """
    e.g. input: "I wake up early in the morning...."
    return: "I wake up early in the morning."   (noly one Period)
    """
    index = list(range(len(text)))
    index.reverse()
    for idx in index:
        if text[idx]!=".":
            break
    return text[:idx+1]+"."


def get_format_train_text(data: dict,mode:str,return_tag=False, text_k="text", subj_k="subj",
                          obj_k="obj", subj_t_k="subj_type", obj_t_k="obj_type",subj_pos = "subj_pos", obj_pos = "obj_pos"):

    

    res = copy.deepcopy(data)
    if mode=='wiki':
        text_formatted = []
        all_prefix = set() 
        index = 0
        for text, obj, subj, obj_t,subj_t,  in tqdm(zip(data[text_k], data[obj_k], data[subj_k],
                                                data[obj_t_k], data[subj_t_k]),total=len(data[text_k])):
            text_f = replace_s(text,obj,obj_prefix(obj, obj_t))
            text_f = replace_s(text_f,subj,subj_prefix(subj, subj_t))
            if assure_replace(text_f)<0:
                print(index,assure_replace(text_f))
            all_prefix.add("<O:{}>".format(obj_t))
            all_prefix.add("<S:{}>".format(subj_t))
            all_prefix.add("</O:{}>".format(obj_t))
            all_prefix.add("</S:{}>".format(subj_t))
            text_formatted.append(text_f)
            index+=1
    elif mode=='tac':
        text_formatted,all_prefix = get_format_train_text_tac(data=data,text_k=text_k, subj_k=subj_k,
                          obj_k=obj_k, subj_t_k=subj_t_k, obj_t_k=obj_t_k,subj_pos = subj_pos, obj_pos = obj_pos)
    else:
        raise error
    
    res['text'] = text_formatted
    all_prefix = list(all_prefix)
    if return_tag:
        return res,all_prefix
    else:
        return res


if __name__ == "__main__":
    text = 'Jefferson J DeBlanc , a World War II fighter pilot who was awarded the Medal of Honor for shooting down five Japanese planes on a single day while running out of fuel , died Nov 22 in Lafayette , La'
    replace_s_tac(text,'Jefferson J DeBlanc',"<'Jefferson J DeBlanc'>")
    
    test_cases = {
        'text':["hi, I went to the high school yesterday.","he went to the university yesterday."],
        'subj':['I','he'],
        'obj':['high school','university'],
        'subj_type':['PER','PER'],
        'obj_type':['LOC','LOC']
    }
    dict,tags = get_format_train_text(test_cases,return_tag=True)
    print(dict)
    print(tags)
    