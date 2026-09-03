# IMDb movie data

`movie_metadata.csv` is the IMDB 5000 Movie Dataset used for Project 4.
It contains 5,043 rows of movie metadata. The application keeps this raw file
unchanged and performs cleaning in `project4/services/movie_data.py`.

The loader:

- extracts the stable IMDb title ID from each movie link;
- removes 124 repeated IMDb entries, leaving 4,919 unique movies;
- removes trailing non-breaking spaces from movie titles;
- retains descriptive metadata for the study interface; and
- builds genre, release-era, language, content-rating, and duration features.

Popularity, financial, review-count, social-media, and IMDb-score fields are
not included in the participant preference vector.

The report records that the file was supplied for the course and cites its
public catalogue listing. Redistribution permissions should be verified with
the dataset provider before using the file outside this coursework submission.
