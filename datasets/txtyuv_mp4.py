def replace_yuv_with_mp4(file_name):
    with open(file_name, 'r') as file:
        contents = file.read()

    new_contents = contents.replace('.yuv', '.mp4')

    with open(file_name, 'w') as file:
        file.write(new_contents)


# 使用函数
replace_yuv_with_mp4('1.txt')
