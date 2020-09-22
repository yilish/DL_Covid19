src_file = open('/tmp/pycharm_project_828/labels.txt', 'r')
dest_file = open('/tmp/pycharm_project_828/changedlabel.txt', 'w')

print('processing...')
lines = src_file.readlines()
for line in lines:
    line = line.strip('\n')
    line = line.rstrip()
    words = line.split(' ')
    dest_file.write(f'{words[0]} ')
    for i in range(1, 11):
        val = float(words[i]) * (2 / (10 - float(words[i])))
        # ai*（2/k-ai）

        dest_file.write(str(format(val, '.3f')))
        dest_file.write(' ')
    dest_file.write('\n')

print('finished!')
src_file.close()
dest_file.close()