# Load the libraries
library(raster)
library(rgdal)
library(parallel)
library(sf)
library(dplyr)
library(data.table)
library(sqldf)
library(doMC)
registerDoMC(35)

main_dir <- "Extracted2/2018"
csv_dir <- "TrainData/TrainingCSVDataCrop/2018/"

# Remove all files in csv_dir
if (length(list.files(csv_dir)) > 0) {
    file.remove(list.files(csv_dir, full.names = T))
}

# Load the shapefile with labels
shp <- readOGR(paste0("shapefiles", "/landcoverLabels2018rev.shp"))
shp_df <- as.data.frame(shp)
shp_df$id <- 1:(nrow(shp_df))

images <- sort(list.files(main_dir, pattern = ".tif"))

#for(i in seq(1, length(images), 1)) {
foreach(i = 1:length(images)) %dopar% {

    print(paste("Processing Image number: ", i, " out of ", length(images)))

    dat <- stack(paste0(main_dir, "/", images[i]))
    file_name <- substr(names(dat)[1], 18, 25)
    #print(dat)
    print(file_name)

    extct <- extract(dat, shp, df = T)
    names(extct) <- c("id", "band1", "band2", "band3")

    final_data <- left_join(extct, shp_df, by = "id")
    final_data["Date"] <- file_name

    print(paste("Saving file: ", file_name, ".csv"))
    write.csv(
      final_data,
      paste0(csv_dir, file_name, ".csv"),
      row.names = T
    )
}