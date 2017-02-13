#!/usr/bin/env python
# -*- coding: utf-8 -*-
import spacy
import csv
import os
import sys
import numpy as np
import pickle

reload(sys)
sys.setdefaultencoding("utf-8")

nlp = spacy.load('en')


# Parses correctly, yay.
# doc2 = nlp(unicode("So Obama used to tell classmates that he was Kenyan royalty and an Indonesian prince http://to.pbs.org/W14iVJÂ  Sounds like his book bio!", errors='ignore'))
# for sent in doc2.sents:
# 	print sent

paragraph = "You can't have Bush.  The last thing we need is another Bush.  Now I made that statement very strongly, and now every one says the last thing--  You know they copied it.  I'll be accused of copying the statement; that's the bad thing.  But I said it.  I was the one that said it first, and I mean it.  The last thing we need is another Bush.  Now, he's totally in favor of Common Core; that's a disaster, that's bad, it should be local and all of that.  But he's totally in favor of Common Core.  He's very, very weak on immigration.  Don't forget--remember his statement--they come for love.  I say, what?  Come for love?  You've got these people coming, half of them are criminals.  They're coming for love?  They're coming for a lot of other reasons, and it's not love.  And when he runs, you got to remember his brother really gave us Obama.  I was never a big fan, but his brother gave us Obama.  'Cause Abraham Lincoln coming home back from the dead could not have won the election because it was going so badly and the economy was just absolutely in shambles that last couple of months.  And then he appointed Justice Roberts.  And Jeb wanted Justice Roberts.  And Justice Roberts basically approved Obamacare in a shocking decision that nobody believes.  So you can't have Jeb Bush.  And he's going to lose aside from that; he's not going to win.  So Mitt and--you just can't have those two.  That's it.  That's it.  It's so simple."
doc1 = nlp(unicode(paragraph))
for sent in doc1.sents:
	# print sent
	for word in sent:
		print word