# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:15:21 2019

@author: 12642
"""

words={"wang","ek","wang","yi","bo"}
word_to_id = dict(zip(words, range(len(words))))
print(word_to_id)
categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
cat_to_id = dict(zip(categories, range(len(categories))))
print(cat_to_id)