# -*- coding: utf-8 -*-

import ast
import re

import argparse

parser = argparse.ArgumentParser(description='Plagiarism_checker')
parser.add_argument('fromd', type=str, help='Input directory to analyze')
parser.add_argument('tod', type=str, help='Output directory for result')
args = parser.parse_args()

"""# Вычисление расстояния Левенштейна"""

def lev(a,b):

  if a==b:
    return 0

  l_a = len(a)
  l_b = len(b)
  
  s1 = [i for i in range(l_a+1)] # Так как для алгоритма нужны только две строки матрицы,
  s2 = [i for i in range(l_b+1)] # Будем использовать два однромерных массива вместо неё


  for i in range(1,l_a+1):

    s2_new = [i for i in range(l_b+1)]
    s2_new[0] = i # массив новой строки с первым элем на 1 больше


    for j in range(1, l_b+1):


      eq_letters = 1     
      if a[i-1] == b[j-1]:
        eq_letters = 0
      
      s2_new[j]=min(s2_new[j-1]+1,s2[j]+1,s2[j-1]+eq_letters)


    s2=s2_new
  
  return s2[-1]

"""# Метрика "похожести" программ"""

def sim(a,b):

  l_a = len(a)
  l_b = len(b)

  ans = (1-lev(a,b)/(max(l_a,l_b)))
  ans = str(float('{:.3f}'.format(ans*100))) + "%"

  return ans

sim("aba","bb")

"""Сначала предобратываем код: Преобразовываем в дерево. Так как, это учитывает удаление комментариев и лишних пробелов, для улучшения точности работы остается удалить import в полученном дереве

Также любое дерево начинается и заканичватеся одинкаово эти части тоже стоит обрезать для повышения точности

# Функция удаления импортов из AST
"""

def removeImports(dump):
# Ищем сначала ImportFrom, затем Import для лучшей работы

  while re.search(r'ImportFrom',dump):
    beg=re.search(r'ImportFrom',dump).span()# Если нашелся импорт ищем начало откуда удалять

    
    cnt=1
    isImport=True
    i=beg[1]+1
    while isImport:#Алгоритм: Ищем скобку закрывающую импорт, а затем удаляем этот блок
      if dump[i]=="(":
        cnt+=1
      elif dump[i]==")":
        cnt-=1
      if cnt==0:
        isImport=False
      i+=1
    
  
    dump=dump[:beg[0]]+dump[i+2:]


  while re.search(r'Import',dump):
    beg=re.search(r'Import',dump).span()# Если нашелся импорт ищем начало откуда удалять

    
    cnt=1
    isImport=True
    i=beg[1]+1  # Конец нахождения Import
    while isImport:
      if dump[i]=="(":
        cnt+=1
      elif dump[i]==")":
        cnt-=1
      if cnt==0:
        isImport=False
      i+=1
    
    
    dump=dump[:beg[0]]+dump[i+2:]#+2 чтобы удалить последний символ и скобку за ним
  #for i in range(len(dump)):
  return dump

#a="Module(body=[Import(names=[alias(name='ast', asname=None)])], type_ignores=[])"
#print("Было: ",a)
#print("Стало: ",removeImports(a)) # Все скобки в дальнейшем удаляться
#Было:  Module(body=[Import(names=[alias(name='ast', asname=None)])], type_ignores=[])
#Стало:  Module(body=[ type_ignores=[])

"""# Чтение списка файлов"""

with open(args.fromd) as f:
  files_list = f.read().split('\n')

for i in range(len(files_list)):
  files_list[i] = files_list[i].split() #Разбиваем список пар файлов в массив

res=open(args.tod,"w")
res.write("Процент совпадения для файлов:\n")

for pare_of_files in files_list:#Для каждой пары считаем процент схожих символов в файлах

  with open(pare_of_files[0]) as f1:
    firstf = f1.read()
  
  with open(pare_of_files[1]) as f2:
    secondf = f2.read()

  #Предрабатываем первый файл

  parsed_fst_file = ast.dump(ast.parse(firstf))# Строим AST

  # Нам интересен только модуль body - Обрежем module и type_ignores
  parsed_fst_file = parsed_fst_file[13:-19]

  parsed_fst_file = removeImports(parsed_fst_file)# Убираем Импорты

  pattern=re.compile(r"[^A-Za-z0-9]")#Оставляем только буквы и цифры

  parsed_fst_file=pattern.sub('',parsed_fst_file)

  #Предрабатываем второй файл

  parsed_scd_file = ast.dump(ast.parse(secondf))# Строим AST

  # Нам интересен только модуль body - Обрежем module и type_ignores
  parsed_scd_file = parsed_scd_file[13:-19]

  parsed_scd_file = removeImports(parsed_scd_file)# Убираем Импорты

  pattern=re.compile(r"[^A-Za-z0-9]")#Оставляем только буквы и цифры

  parsed_scd_file=pattern.sub('',parsed_scd_file)

  #Считаем степень совпадения кодов

  res.write(pare_of_files[0]+" и "+pare_of_files[1]+" - "+str(sim(parsed_fst_file,parsed_scd_file))+"\n")

res.close()