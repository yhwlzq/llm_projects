def transformSentence(sentence):
    if len(sentence) == 0:
        return ""
    result = [sentence[0]]
    for i in range(1, len(sentence)):
        char = sentence[i]
        if not char.isalnum():
            result.append(char)
            continue
            
        prev_char = result[-1]
        # 获取用于比较的字符（字母转换为小写，数字保持不变）
        comp_prev = prev_char.lower() if prev_char.isalpha() else prev_char
        comp_char = char.lower() if char.isalpha() else char
        
        # 比较字符（基于字母顺序或ASCII值）
        if comp_prev < comp_char:
            new_char = char.upper() if char.isalpha() else char
            result.append(new_char)
        elif comp_prev > comp_char:
            new_char = char.lower() if char.isalpha() else char
            result.append(new_char)
        else:
            result.append(char)
    return "".join(result)
    # return "".join(result)
            
    if __name__ == '__main__':
        fptr = open(os.environ['OUTPUT_PATH'], 'w')

        sentence = input()

        result = transformSentence(sentence)

        fptr.write(result + '\n')

        fptr.close()
