# Creation of a csv for classes <-> Ecotaxa's IDs match.

library(ecotaxar) # https://github.com/ecotaxa/ecotaxar
library(tidyverse)

db <- db_connect_ecotaxa() # connect to Ecotaxa
taxo <- tbl(db, "taxonomy") %>% collect() # get taxo info
db_disconnect_ecotaxa(db) # disconnect

# clean a bit the data
taxo <- taxo %>% select(id, display_name) %>% distinct()

# save data
write_csv(taxo, 'id_classes_ecotaxa.csv')
