s = '<mask> was born in 1927 in the United States of America and is a registered nurse.'
s_split = s.split('is a ')
print(s_split[-1].split('.')[0].replace(' ', '_'))