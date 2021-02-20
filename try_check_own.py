import numpy as np

sen = ' i am a good   boy '
words = sen.split(' ')
words = [w for w in words if len(w)>0]
words = ' '.join(words)
print("1"+words+'1')