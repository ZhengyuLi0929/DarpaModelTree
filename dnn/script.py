import os


# infoIDs_twitter = [u'guaido/illegitimate', u'protests', u'maduro/events/anti', u'assembly/legitimate', u'maduro/russia_support', u'international/military', u'arrests', u'international/respect_sovereignty', u'maduro/cuba_support', u'international/aid_rejected', u'violence', u'maduro/narco', u'other/planned_coup', u'crisis', u'other/restore_democracy', u'guaido/us_support', u'international/break_us_relations', u'arrests/opposition/media', u'other/anti_socialism', u'maduro/legitimate', u'international/emigration', u'crisis/looting', u'other/chavez/anti', u'maduro/dictator', u'arrests/opposition', u'other/request_observers', u'international/aid', u'maduro/illegitimate', u'maduro/events/pro', u'arrests/opposition/protesters', u'guaido/legitimate', u'crisis/lack_essentials', u'maduro/events', u'international/us_sanctions', u'maduro/illegitimate/international', u'guaido/legitimate/international', u'violence/against_opposition', u'violence/against_opposition/protesters', u'violence/against_maduro', u'maduro/legitimate/international', u'other/media_bias', u'other/chavez/pro', u'other/chavez', u'other/censorship_outage', u'military', u'military/desertions', u'assembly/illegitimate']
# infoIDs_youtube = [u'protests', u'assembly/legitimate', u'maduro/events', u'international/military', u'violence/against_opposition', u'international/respect_sovereignty', u'maduro/cuba_support', u'international/aid_rejected', u'maduro/narco', u'other/planned_coup', u'crisis', u'guaido/us_support', u'other/restore_democracy', u'other/anti_socialism', u'maduro/legitimate', u'international/emigration', u'other/chavez/anti', u'maduro/dictator', u'arrests/opposition', u'international/aid', u'maduro/illegitimate', u'maduro/events/pro', u'arrests/opposition/protesters', u'guaido/legitimate', u'crisis/lack_essentials', u'international/us_sanctions', u'guaido/legitimate/international', u'arrests', u'other/chavez/pro', u'violence', u'maduro/legitimate/international', u'other/media_bias', u'violence/against_opposition/protesters', u'other/chavez', u'other/censorship_outage', u'military', u'military/desertions']

# for infoID in infoIDs_twitter:
# 	os.system("python dnn.py twitter "+infoID)

# for infoID in infoIDs_youtube:
# 	os.system("python dnn.py youtube "+infoID)



os.system("python dnn.py twitter arrests")