def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

a = remove_prefix(prefix='abce', text='abceecba')
print(a)