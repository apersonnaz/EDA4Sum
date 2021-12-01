from app.pipelines.tools.dataPreparation import prepare_data
import pandas as pd

# columns = ["dm-authors.author_id", "dm-authors.gender", "dm-authors.seniority", "dm-authors.nb_publi",
#            "dm-authors.pub_rate", "dm-authors.first_author_por", "dm-authors.avg_coauthor",
#            "dm-authors.country", "dm-authors.CIKM", "dm-authors.ICDE", "dm-authors.ICWSM", "dm-authors.IEEE", "dm-authors.RecSys",
#            "dm-authors.SIGIR", "dm-authors.SIGMOD", "dm-authors.VLDB", "dm-authors.WSDM", "dm-authors.WWW"]

# prepare_data(database_name="dm-authors", initial_collection_names=["dm-authors"], discrete_categories_count=5,
#              id_attribute_name="dm-authors.author_id", index_build_process_count=3, data_folder="./app/data/", build_groups=True,
#              build_selectivity_index=False, build_index=False, min_group_size=10, exploration_columns=columns)

# prepare_data(database_name="sdss", initial_collection_names=["galaxies"], discrete_categories_count=10,
#              id_attribute_name="galaxies.objID", index_build_process_count=3, data_folder="./app/data/", build_groups=True,
#              build_selectivity_index=False, build_index=False, min_group_size=10, exploration_columns=["galaxies.objID", "galaxies.u", "galaxies.g", "galaxies.r",
#                                                                                                        "galaxies.i", "galaxies.z", "galaxies.petroRad_r", "galaxies.redshift"])


columns = ["tracks.track_id", "tracks.popularity", "tracks.acousticness", "tracks.danceability",
           "tracks.duration_ms", "tracks.energy", "tracks.instrumentalness",
           "tracks.liveness", "tracks.loudness", "tracks.speechiness", "tracks.tempo", "tracks.valence", ]
# "tracks.key",
#    s"tracks.mode"]

# prepare_data(database_name="spotify", initial_collection_names=["tracks"], discrete_categories_count=10,
#              id_attribute_name="tracks.track_id", index_build_process_count=3, data_folder="./app/data/", build_groups=True,
#              build_selectivity_index=False, build_index=False, min_group_size=10, exploration_columns=columns)


ds_name = "tracks_23000"
columns = [f"{ds_name}.track_id", f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
           f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
           f"{ds_name}.liveness", f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
prepare_data(database_name="spotify", initial_collection_names=[ds_name], discrete_categories_count=10,
             id_attribute_name=f"{ds_name}.track_id", index_build_process_count=3, data_folder="./app/data/", build_groups=True,
             build_selectivity_index=False, build_index=False, min_group_size=20, exploration_columns=columns)

ds_name = "tracks_115000"
columns = [f"{ds_name}.track_id", f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
           f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
           f"{ds_name}.liveness", f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
prepare_data(database_name="spotify", initial_collection_names=[ds_name], discrete_categories_count=10,
             id_attribute_name=f"{ds_name}.track_id", index_build_process_count=3, data_folder="./app/data/", build_groups=True,
             build_selectivity_index=False, build_index=False, min_group_size=20, exploration_columns=columns)

ds_name = "tracks_5b"
columns = [f"{ds_name}.track_id", f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
           f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
           f"{ds_name}.liveness", f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
prepare_data(database_name="spotify", initial_collection_names=[ds_name], discrete_categories_count=5,
             id_attribute_name=f"{ds_name}.track_id", index_build_process_count=3, data_folder="./app/data/", build_groups=True,
             build_selectivity_index=False, build_index=False, min_group_size=20, exploration_columns=columns)

ds_name = "tracks_20b"
columns = [f"{ds_name}.track_id", f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
           f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness",
           f"{ds_name}.liveness", f"{ds_name}.loudness", f"{ds_name}.speechiness", f"{ds_name}.tempo", f"{ds_name}.valence", ]
prepare_data(database_name="spotify", initial_collection_names=[ds_name], discrete_categories_count=20,
             id_attribute_name=f"{ds_name}.track_id", index_build_process_count=3, data_folder="./app/data/", build_groups=True,
             build_selectivity_index=False, build_index=False, min_group_size=20, exploration_columns=columns)

ds_name = "tracks_7c"
columns = [f"{ds_name}.track_id", f"{ds_name}.popularity", f"{ds_name}.acousticness", f"{ds_name}.danceability",
           f"{ds_name}.duration_ms", f"{ds_name}.energy", f"{ds_name}.instrumentalness", f"{ds_name}.liveness"]
prepare_data(database_name="spotify", initial_collection_names=[ds_name], discrete_categories_count=10,
             id_attribute_name=f"{ds_name}.track_id", index_build_process_count=3, data_folder="./app/data/", build_groups=True,
             build_selectivity_index=False, build_index=False, min_group_size=20, exploration_columns=columns)


ds_name = "tracks_3c"
columns = [f"{ds_name}.track_id", f"{ds_name}.popularity",
           f"{ds_name}.acousticness", f"{ds_name}.danceability"]
prepare_data(database_name="spotify", initial_collection_names=[ds_name], discrete_categories_count=10,
             id_attribute_name=f"{ds_name}.track_id", index_build_process_count=3, data_folder="./app/data/", build_groups=True,
             build_selectivity_index=False, build_index=False, min_group_size=20, exploration_columns=columns)

#   s.class = 'GALAXY'
#   AND s.z between 0.11 AND 0.36
#   AND p.r >= 18 and p.r <= 20.5
#   AND p.petrorad_r < 2
#   And p.u-p.r <= 2.5 and p.r-p.i <= 0.2 and p.r-p.z <= 0.5
#   And p.g-p.r >= p.r-p.i+0.5 and p.u-p.r  >= 2.5*(p.r-p.z)
#   AND oiii_5007_eqw < -100
#   AND h_beta_eqw < -50
