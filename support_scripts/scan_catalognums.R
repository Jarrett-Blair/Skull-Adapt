getAllFiles <- function(directory) {
  # List all files recursively in the directory
  files <- list.files(directory, recursive = TRUE, full.names = TRUE)
  return(files)
}

# Replace 'path_to_your_folder' with the actual path to your folder
folder_path <- "C:/Users/blair/OneDrive - UBC/Skulls/traintex"

# Call the function to get all file names
all_files <- getAllFiles(folder_path)

all_files = basename(all_files)

all_files = gsub("Z-", "Z0", all_files)


extractInfo <- function(file_names) {
  # Split each filename by "-"
  components <- strsplit(file_names, "-")
  
  # Extract the desired component
  extracted_info <- sapply(components, function(x) {
    if (length(x) >= 3) {
      return(x[3])
    } else {
      return(NA)
    }
  })
  
  return(extracted_info)
}


spec_nums <- extractInfo(all_files)

spec_nums = unique(spec_nums)
spec_nums = na.omit(spec_nums)

spec_nums <- gsub("^0+", "", spec_nums)


gbif_idx = which(catalogNumbers %in% spec_nums)

# Plus sum(spec_nums %!in% catalogNumbers)
table(gbif$sex[gbif_idx])






