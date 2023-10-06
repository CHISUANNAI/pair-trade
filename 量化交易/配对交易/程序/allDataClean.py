import shutil

path = '..\\result'
try:
    shutil.rmtree(path)
    print("É¾³ý³É¹¦")
except:
    print("É¾³ýÊ§°Ü")
