
how to use these notebooks and for which purpose

---------------------------------------------------------------------------------------------------------------------

get_wikipedia_language.ipynb: use wikipedia API to grab languages for each species

handle_synonyms.ipynb: clean, verify, make small stat and aggregate synonyms from andrews and wikipedia languages (todo: pdf book etc) 

get_flickr_data.ipynb: using flickr API (first make your own keys that you should put at the begining of the script under API parameter section. To do them follow: http://joequery.me/code/flickr-api-image-search-python/) to collect images using the species name with their synonyms and their language translations.

get_herpmapper_data.ipynb: using andrew csv file with the url we download image using the species name with their synonyms and their language translations.

get_inaturalist_data.ipynb: using andrew csv file with the url we download image using the species name with their synonyms and their language translations.

aggregated_all_datasource_for_dl.ipynb: to aggregate all the images info from various data source and create adequate csv file fro crowdai challenge.

all_images_stat.ipynb: create plot of any dataframe with image info .Choose at begining which one to use. e.g. df_all_datasource, df_crowdai, df_crowdai_test, df_crowdai_train.

frompdf2text.ipynb: try to structure the information from a pdf book to gather more synonyms

fromscannpdf2mage.ipynb: get images from scan pdf books


--------------
data we miss:
herpmapper: latitude, longitude, license
snapp: datetaken, latitude, longitude,

note:
snapp: id use aso the jpg or png' format as it was done like this by andrew and that the endpoint change. (i..e. saved_img_id='id, and not typically 'snapp_'+x['species']+'_'+str(x['id'])+".png"






