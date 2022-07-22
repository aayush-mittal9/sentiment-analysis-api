file_name_and_text = {}
for path,dirs,files in os.walk(path):
    for file in files:
        if file.endswith('.html'):
            fullname = os.path.join(path,file)
            with open(fullname, "r",encoding='utf-8') as target_file:
                file_name_and_text[file] = target_file.read()
df = (pd.DataFrame.from_dict(file_name_and_text, orient='index')
             .reset_index().rename(index = str, columns = {'index': 'file_name', 0: 'text'}))
df.head()
