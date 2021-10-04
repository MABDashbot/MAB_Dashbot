import csv

root = '/home/sandrine/DashBot/bandits-and-applications/code/DashBot/implem/dataset/MovieLens'

with open(f'{root}/raw_movielens.csv', newline='') as csvfile:
	attributes = csvfile.readline().strip().split('|')
	# print(fieldnames)
	# ['user_id', 'movie_id', 'rating', 't', 'age', 'gender', 'occupation', 'zip_code', 'movie_title', 'release_date', 'video_release_date',
	# 'IMDb_URL', 'g_unknown', 'g_Action', 'g_Adventure', 'g_Animation', 'g_Childrens', 'g_Comedy', 'g_Crime', 'g_Documentary', 'g_Drama',
	# 'g_Fantasy', 'g_Film_Noir', 'g_Horror', 'g_Musical', 'g_Mystery', 'g_Romance', 'g_Sci_Fi', 'g_Thriller', 'g_War', 'g_Western']
	attributes.remove('IMDb_URL')
	attributes.remove('movie_id')

with open(f'{root}/raw_movielens.csv', newline='', encoding = "ISO-8859-1") as inputfile:
	reader = csv.DictReader(inputfile, delimiter='|')

	with open(f'{root}/MovieLens.csv', 'w', newline='') as outputfile:
		writer = csv.DictWriter(outputfile, fieldnames=attributes)

		writer.writeheader()

		for input_row in reader:

			output_row = dict()
			for att in attributes:
				output_row[att] = input_row[att]

			x = input_row['movie_title'].strip().split(' (')

			movie_title = ' ('.join(x[:-1])
			release_date = x[-1][:-1]
			if len(release_date) != 4:
				x = movie_title.strip().split(' (')
				movie_title = ' ('.join(x[:-1])
				release_date = x[-1][:-1]

			if release_date == 'unknown':
				release_date = None

			if movie_title[-5:] == ', The':
				x = movie_title.split(', ')
				movie_title = 'The ' + x[0]

			elif movie_title[-3:] == ', A':
				x = movie_title.split(', ')
				movie_title = 'A ' + x[0]

			output_row['movie_title'] = movie_title
			output_row['release_date'] = release_date




			writer.writerow(output_row)
		




# with open(f'{root}/SERAC_raw.csv', newline='') as input_file:
# 	reader = csv.DictReader(input_file)
# 	for row in reader:
# 		cleaned_row = dict()

# 		# activity
# 		if 'activity' in selected_attributes:
# 			val = row["Activités"]
# 			if val == "VTT,":
# 				cleaned_row['activity'] = 'mountain bike'
# 			if val == "parapente," :
# 				cleaned_row['activity'] = 'paragliding'
# 			if val == "cascade de glace," : 
# 				cleaned_row['activity'] = 'ice climbing'
# 			if val == "randonnée," : 
# 				cleaned_row['activity'] = 'hiking'
# 			if val in ["escalade,randonnée,", "escalade,"] : 
# 				cleaned_row['activity'] = 'climbing'
# 			if val in [ 'ski de randonnée,', 'ski de randonnée,raquettes,' , 'raquettes,ski de randonnée,'] : 
# 				cleaned_row['activity'] = 'backcountry'
# 			if val in ["rocher haute-montagne," , "rocher haute-montagne,escalade,", "rocher haute-montagne,randonnée,"] : 
# 				cleaned_row['activity'] = 'rock mountaineering'
# 			if val in ['neige glace mixte,', "randonnée,neige glace mixte,", 'ski de randonnée,neige glace mixte,'] : 
# 				cleaned_row['activity'] = 'snow/ice/mixed'
# 			if val in ["rocher haute-montagne,neige glace mixte,", "neige glace mixte,rocher haute-montagne,"] : 
# 				cleaned_row['activity'] = 'mountaineering'

# 		# location
# 		if 'location' in selected_attributes:
# 			cleaned_row['location'] = row["Localisation"]

# 		# altitude
# 		if 'altitude' in selected_attributes:
# 			cleaned_row['altitude'] = row["Altitude"]

# 		# date
# 		if 'date' in selected_attributes:
# 			cleaned_row['date'] = row["Date"]

# 		#  N participant(s)
# 		if 'N participant(s)' in selected_attributes:
# 			cleaned_row['N participant(s)'] = row["Nombre de participants"]

# 		# N victims
# 		if 'N victim(s)' in selected_attributes:
# 			cleaned_row['N victim(s)'] = row["Nombre de personnes touchées"]

# 		# slope
# 		if 'slope' in selected_attributes:
# 			cleaned_row['slope'] = row["Pente de la zone de départ"]

# 		# age
# 		if 'age' in selected_attributes:
# 			cleaned_row['age'] = row["Âge"]

# 		# country
# 		val = row["Régions"]
# 		val = val.split(',')
# 		val = [v for v in val if len(v) > 0]
# 		if len(val) > 0:
# 			if val[0] == 'Provence':
# 				if 'country' in selected_attributes:
# 					cleaned_row['country'] = 'France'
# 				if 'region' in selected_attributes:
# 					cleaned_row['region'] = val[0]
# 				if 'massif' in selected_attributes:
# 					cleaned_row['massif'] = val[1]
# 			else:
# 				if 'country' in selected_attributes:
# 					if val[0] == 'France':
# 						cleaned_row['country'] = 'France'
# 					if val[0] == 'Suisse':
# 						cleaned_row['country'] = 'Switzerland'
# 					if val[0] == 'Italie':
# 						cleaned_row['country'] = 'Italy'
# 					if val[0] == 'Espagne':
# 						cleaned_row['country'] = 'Spain'
# 					if val[0] == 'Royaume-Uni':
# 						cleaned_row['country'] = 'UK'
# 					if val[0] == 'Belgique':
# 						cleaned_row['country'] = 'Belgium'
# 					if val[0] == 'Andorre':
# 						cleaned_row['country'] = 'Andorra'
# 					if val[0] == 'Nouvelle-Zélande':
# 						cleaned_row['country'] = 'New-Zealand'
# 					if val[0] == 'Éthiopie':
# 						cleaned_row['country'] = 'Ethiopia'
# 					if val[0] == 'Cuba':
# 						cleaned_row['country'] = 'Cuba'
# 					if val[0] == 'Argentine':
# 						cleaned_row['country'] = 'Argentina'
# 					if val[0] == 'Kirghizstan':
# 						cleaned_row['country'] = 'Kirghizstan'
# 					if val[0] == 'Colombie':
# 						cleaned_row['country'] = 'Colombia'
# 					if val[0] == 'Chine':
# 						cleaned_row['country'] = 'China'
# 					if val[0] == 'Laos':
# 						cleaned_row['country'] = 'Laos'
# 					if val[0] == 'Autriche':
# 						cleaned_row['country'] = 'Austria'

# 				# region / massif
# 				if val[0] in ['Italie', 'Espagne', 'Autriche']:
# 					if 'region' in selected_attributes:
# 						cleaned_row['region'] = val[2]
# 					if val[1] in ['Valais W - Alpes Pennines W', 'Valais E - Alpes Pennines E']:
# 						if 'massif' in selected_attributes:
# 							cleaned_row['massif'] = 'Valais - Alpes Pennines'
# 					else:
# 						if 'massif' in selected_attributes:
# 							cleaned_row['massif'] = val[1]
# 				else:
# 					if len(val) > 1:
# 						if 'region' in selected_attributes:
# 							cleaned_row['region'] = val[1]
# 						if len(val) == 3:
# 							if 'massif' in selected_attributes:
# 								if val[2] in ['Valais W - Alpes Pennines W', 'Valais E - Alpes Pennines E']:
# 									cleaned_row['massif'] = 'Valais - Alpes Pennines'
# 								else:
# 									cleaned_row['massif'] = val[2]

# 		# event/cause
# 		val = row["Type d\'évènement"]
# 		if 'event' in selected_attributes:
# 			if val in ['chute encordé,autre,', 'chute encordé,', 'défaillance physique,chute encordé,']:
# 				# 149 : cause = défaillance physique
# 				cleaned_row['event'] = 'roped fall'
# 			if val in ["chute d'une personne,", "chute d'une personne,chute encordé,", "chute encordé,chute d'une personne,", "chute d'une personne,autre,", "défaillance physique,chute encordé,chute d'une personne,", "défaillance physique,chute d'une personne", "chute d'une personne,défaillance physique,"]:
# 				cleaned_row['event'] = 'fall'
# 			if val == "défaillance physique,":
# 				# 222 & 301 : cause = défaillance physique
# 				cleaned_row['event'] = 'injury'
# 		if 'cause' in selected_attributes:
# 			if val == 'chute de pierres,':
# 				cleaned_row['cause'] = 'falling rocks'
# 			if val == 'chute de glace,':
# 				cleaned_row['cause'] = 'falling ice'
# 			if val in ['avalanche,', 'avalanche,chute de pierres,']:
# 				cleaned_row['cause'] = 'avalanche'
# 			if val == 'foudre,':
# 				cleaned_row['cause'] = 'lightning'
# 		if val == 'défaillance physique,avalanche,':
# 			if 'cause' in selected_attributes:
# 				cleaned_row['cause'] = 'avalanche'
# 			if 'event' in selected_attributes:
# 				cleaned_row['event'] = 'injury'
# 		if val in ["défaillance physique,chute d'une personne,chute de pierres,", "chute de pierres,chute d'une personne,", "chute de pierres,chute encordé,chute d'une personne,", "chute de pierres,chute d'une personne,chute encordé,", "chute d'une personne,chute de pierres,"]:
# 			if 'event' in selected_attributes:
# 				cleaned_row['event'] = 'fall'
# 			if 'cause' in selected_attributes:
# 				cleaned_row['cause'] = 'falling rocks'
# 		if val == "chute en crevasse,chute d'une personne,chute encordé,":
# 			if 'cause' in selected_attributes:
# 				cleaned_row['cause'] = 'crevasse'
# 			if 'event' in selected_attributes:
# 				cleaned_row['event'] = 'fall'
# 		if val in ["chute encordé,chute de pierres,", "chute de pierres,chute encordé,", "chute encordé,chute de pierres,autre,"]:
# 			if 'cause' in selected_attributes:
# 				cleaned_row['cause'] = 'falling rocks'
# 			if 'event' in selected_attributes:
# 				cleaned_row['event'] = 'roped fall'
# 		if val == "chute en crevasse,":
# 			# 372 : cause = avalanche, event = chute
# 			if 'cause' in selected_attributes:
# 				cleaned_row['cause'] = 'crevasse'
# 			if 'event' in selected_attributes:
# 				cleaned_row['event'] = 'roped fall'
# 		if val == "chute encordé,chute en crevasse,":
# 			if 'cause' in selected_attributes:
# 				cleaned_row['cause'] = 'crevasse'
# 			if 'event' in selected_attributes:
# 				cleaned_row['event'] = 'roped fall'
# 		if val == "avalanche,chute d'une personne,":
# 			if 'cause' in selected_attributes:
# 				cleaned_row['cause'] = 'avalanche'
# 			if 'event' in selected_attributes:
# 				cleaned_row['event'] = 'fall'
# 		if val == "chute d'une personne,chute de glace,":
# 			if 'cause' in selected_attributes:
# 				cleaned_row['cause'] = 'falling ice'
# 			if 'event' in selected_attributes:
# 				cleaned_row['event'] = 'fall'
		


# 		# rescue
# 		if 'rescue' in selected_attributes:
# 			val = row["Intervention des services de secours"]
# 			if val == 'oui':
# 				cleaned_row['rescue'] = True
# 			if val == 'non':
# 				cleaned_row['rescue'] = False

# 		# severity
# 		if 'severity' in selected_attributes:
# 			val = row["Gravité"]
# 			if val =='pas de blessure' : 
# 				cleaned_row['severity'] = 0 # '0'
# 			if val == 'De 1 à 3 jours' : 
# 				cleaned_row['severity'] = 0.02 # '1-3'
# 			if val == 'De 4 jours à 1 mois' :
# 				cleaned_row['severity'] = 0.17 # '4-30'
# 			if val == 'De 1 à 3 mois' :
# 				cleaned_row['severity'] = 0.6 # '30-90'
# 			if val == 'supérieur à 3 mois' :
# 				cleaned_row['severity'] = 0.9 # '>90'

# 		# BRA
# 		if 'BRA' in selected_attributes:
# 			val = row["Niveau de risque d\'avalanche" ]
# 			try:
# 				cleaned_row['BRA'] = int(val.split(' - ')[0])
# 			except ValueError:
# 				pass

# 		# gender
# 		if 'gender' in selected_attributes:
# 			val = row["Sexe"]
# 			if val == 'H':
# 				cleaned_row['gender'] = 'M'
# 			else:
# 				cleaned_row['gender'] = val

# 		# level
# 		if 'level' in selected_attributes:
# 			val = row["Niveau de pratique"]
# 			if val == 'expert':
# 				cleaned_row['level'] = 'expert'
# 			if val == 'débrouillé':
# 				cleaned_row['level'] = 'initiated'
# 			if val == 'autonome':
# 				cleaned_row['level'] = 'self-sufficient'
# 			if val == 'non autonome':
# 				cleaned_row['level'] = 'novice'

# 		# frequency
# 		if 'frequency' in selected_attributes:
# 			val = row["Fréquence de pratique dans l\'activité"]
# 			if val == "moins d'1 fois par an" :
# 				cleaned_row['frequency'] =  '<1'
# 			if val == "moins d'1 fois par mois" : 
# 				cleaned_row['frequency'] = '1-10'
# 			if val == '1 fois par mois' : 
# 				cleaned_row['frequency'] = '10-20'
# 			if val == '2 à 3 fois par mois' : 
# 				cleaned_row['frequency'] = '20-50'
# 			if val == '1 à 2 fois par semaine' : 
# 				cleaned_row['frequency'] = '50-150'
# 			if val == 'au moins 3 fois par semaine' : 
# 				cleaned_row['frequency'] = '>150'

# 		with open(f'{root}/SERAC.csv', 'a', newline='') as output_file:
# 			writer = csv.DictWriter(output_file, fieldnames=selected_attributes)
# 			writer.writerow(cleaned_row)

# values = dict()
# with open(f'{root}/SERAC.csv', newline='') as csvfile:
# 	for col in selected_attributes:
# 		values[col] = set()
# 	reader = csv.DictReader(csvfile)
# 	for row in reader:
# 		for col in selected_attributes:
# 			values[col].add(row[col])		
# with open(f'{root}/SERAC_values.csv', 'w') as file:
# 	for col, vals in values.items():
# 		file.write('\n///////////////////////////\n' + col + '\n////////////////////////////\n')
# 		for val in vals:
# 			file.write(val + '\n')







