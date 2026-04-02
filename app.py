import os
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Netflix Content Analytics",
    page_icon="🎬",
    layout="wide",
)

DATA_FILE = "netflix_titles.csv"

STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "have", "are", "not",
    "you", "your", "but", "into", "from", "about", "also", "their", "more",
    "will", "which", "what", "when", "where", "who", "than", "all", "there",
    "most", "film", "series", "show", "movie", "stories", "story", "drama",
    "family", "life", "based", "true", "new", "one", "two", "three", "four",
    "five", "six", "seven", "his", "her", "its", "has", "had",
}


def load_data(path: str):
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]

    for text_col in ["director", "cast", "country", "listed_in", "rating", "type", "title", "description"]:
        if text_col in df.columns:
            df[text_col] = df[text_col].fillna("Unknown").astype(str)
        else:
            df[text_col] = "Unknown"

    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    else:
        df["date_added"] = pd.NaT

    df["added_year"] = df["date_added"].dt.year.fillna(0).astype(int)
    df["added_month"] = df["date_added"].dt.month_name().fillna("Unknown")

    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
        df["release_year"] = df["release_year"].fillna(0).astype(int)
    else:
        df["release_year"] = 0

    if "duration" in df.columns:
        df["duration"] = df["duration"].fillna("Unknown").astype(str)
    else:
        df["duration"] = "Unknown"

    def parse_duration(value: str) -> int | None:
        if not isinstance(value, str):
            return None
        digits = re.findall(r"(\d+)", value)
        return int(digits[0]) if digits else None

    df["duration_int"] = df["duration"].apply(parse_duration)

    df["genre_list"] = (
        df["listed_in"]
        .astype(str)
        .apply(lambda value: [item.strip() for item in value.split(",") if item.strip()])
        .apply(lambda items: items if items else ["Unknown"])
    )

    df["country_list"] = (
        df["country"]
        .astype(str)
        .apply(lambda value: [item.strip() for item in value.split(",") if item.strip()])
        .apply(lambda items: items if items else ["Unknown"])
    )

    df["cast_list"] = (
        df["cast"]
        .astype(str)
        .apply(lambda value: [item.strip() for item in value.split(",") if item.strip()])
        .apply(lambda items: items if items else ["Unknown"])
    )

    return df


@st.cache_data
def load_cached_data():
    return load_data(DATA_FILE)


def filter_dataset(df: pd.DataFrame, selected_types, selected_countries, selected_years, selected_ratings, selected_genres, search_text):
    filtered = df.copy()

    if selected_types:
        filtered = filtered[filtered["type"].isin(selected_types)]

    if selected_countries:
        filtered = filtered[filtered["country_list"].apply(lambda countries: any(country in selected_countries for country in countries))]

    if selected_years:
        filtered = filtered[filtered["release_year"].isin(selected_years)]

    if selected_ratings:
        filtered = filtered[filtered["rating"].isin(selected_ratings)]

    if selected_genres:
        filtered = filtered[filtered["genre_list"].apply(lambda genres: any(genre in selected_genres for genre in genres))]

    if search_text:
        search_lower = search_text.lower()
        filtered = filtered[
            filtered["title"].str.lower().str.contains(search_lower, na=False)
            | filtered["description"].str.lower().str.contains(search_lower, na=False)
            | filtered["director"].str.lower().str.contains(search_lower, na=False)
            | filtered["cast"].str.lower().str.contains(search_lower, na=False)
        ]

    return filtered


def get_top_n(series: pd.Series, limit: int = 10):
    return (
        series.value_counts()
        .head(limit)
        .rename_axis("label")
        .reset_index(name="count")
    )


def extract_keyword_counts(descriptions: pd.Series, top_n: int = 20) -> pd.DataFrame:
    text = " ".join(descriptions.dropna().astype(str).str.lower())
    tokens = re.findall(r"\b[a-z]{3,}\b", text)
    tokens = [token for token in tokens if token not in STOP_WORDS]
    counts = Counter(tokens)
    return pd.DataFrame(counts.most_common(top_n), columns=["keyword", "count"])


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_dashboard():
    st.title("Netflix Content Analytics Dashboard")
    st.markdown(
        "Explore Netflix titles with modern BI-style analytics, dynamic filters, and interactive Plotly charts."
    )

    df = load_cached_data()
    if df is None:
        st.error(f"Could not find dataset: `{DATA_FILE}`.")
        st.info(
            "Place `netflix_titles.csv` in the project folder before running this app."
        )
        return

    st.sidebar.header("Filters")

    type_options = sorted(df["type"].dropna().unique())
    selected_types = st.sidebar.multiselect("Content Type", type_options, default=type_options)

    all_countries = sorted({country for countries in df["country_list"] for country in countries if country})
    selected_countries = st.sidebar.multiselect("Country", all_countries, default=all_countries)

    year_options = sorted(df.loc[df["release_year"] > 0, "release_year"].unique())
    selected_years = st.sidebar.multiselect("Release Year", year_options, default=year_options)

    rating_options = sorted(df["rating"].replace("", "Unknown").unique())
    selected_ratings = st.sidebar.multiselect("Rating", rating_options, default=rating_options)

    genre_options = sorted({genre for genres in df["genre_list"] for genre in genres if genre})
    selected_genres = st.sidebar.multiselect("Genre", genre_options, default=genre_options)

    search_text = st.sidebar.text_input("Search title, director, cast or description")

    theme = st.sidebar.selectbox("Chart theme", ["plotly_dark", "plotly_white", "seaborn"], index=0)

    if st.sidebar.button("Reset filters"):
        st.experimental_rerun()

    filtered = filter_dataset(
        df,
        selected_types,
        selected_countries,
        selected_years,
        selected_ratings,
        selected_genres,
        search_text,
    )

    if filtered.empty:
        st.warning("No titles match the selected filters. Adjust or clear filters to see results.")
        return

    total_titles = len(filtered)
    total_movies = int((filtered["type"] == "Movie").sum())
    total_tv = int((filtered["type"] == "TV Show").sum())
    total_countries = len({country for countries in filtered["country_list"] for country in countries if country})

    genre_counts = Counter({genre for genres in filtered["genre_list"] for genre in genres if genre})
    most_common_genre = max(genre_counts, key=genre_counts.get) if genre_counts else "Unknown"
    most_common_rating = filtered["rating"].mode().iloc[0] if not filtered["rating"].mode().empty else "Unknown"
    average_duration = filtered["duration_int"].dropna().mean()
    average_duration_label = f"{average_duration:.0f} min" if average_duration else "Unknown"

    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Total Titles", f"{total_titles:,}")
    kpi_cols[1].metric("Total Movies", f"{total_movies:,}")
    kpi_cols[2].metric("Total TV Shows", f"{total_tv:,}")
    kpi_cols[3].metric("Producing Countries", f"{total_countries:,}")
    kpi_cols[4].metric("Top Genre", most_common_genre)
    kpi_cols[5].metric("Top Rating", most_common_rating)

    st.markdown("---")

    with st.expander("Download filtered dataset"):
        st.download_button(
            label="Download filtered titles",
            data=to_csv_bytes(filtered),
            file_name="netflix_filtered_titles.csv",
            mime="text/csv",
        )

    if st.checkbox("Show filtered table", value=False):
        st.dataframe(
            filtered[
                ["show_id", "type", "title", "director", "country", "release_year", "rating", "duration", "listed_in", "date_added"]
            ].sort_values(["release_year", "title"], ascending=[False, True]),
            use_container_width=True,
        )

    with st.container():
        col1, col2 = st.columns([1, 1])
        type_fig = px.pie(
            filtered,
            names="type",
            title="Movie vs TV Show Distribution",
            hole=0.4,
            template=theme,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        year_fig = px.line(
            filtered.groupby("release_year").size().reset_index(name="titles"),
            x="release_year",
            y="titles",
            markers=True,
            title="Titles Released by Year",
            labels={"release_year": "Release Year", "titles": "Titles"},
            template=theme,
        )
        col1.plotly_chart(type_fig, use_container_width=True)
        col2.plotly_chart(year_fig, use_container_width=True)

    with st.container():
        col1, col2 = st.columns([1, 1])
        country_df = get_top_n(pd.Series({country: sum(country in countries for countries in filtered["country_list"]) for country in all_countries}))
        country_fig = px.bar(
            country_df,
            x="count",
            y="label",
            orientation="h",
            title="Top 10 Countries Producing Netflix Content",
            labels={"count": "Titles", "label": "Country"},
            template=theme,
        )

        genre_df = pd.DataFrame(
            Counter({genre: sum(genre in genres for genres in filtered["genre_list"]) for genre in genre_options}).most_common(15),
            columns=["genre", "count"],
        )
        genre_fig = px.bar(
            genre_df,
            x="count",
            y="genre",
            orientation="h",
            title="Top Genres",
            labels={"count": "Titles", "genre": "Genre"},
            template=theme,
        )

        col1.plotly_chart(country_fig, use_container_width=True)
        col2.plotly_chart(genre_fig, use_container_width=True)

    with st.container():
        col1, col2 = st.columns([1, 1])
        rating_df = get_top_n(filtered["rating"].replace("", "Unknown"))
        rating_fig = px.bar(
            rating_df,
            x="count",
            y="label",
            orientation="h",
            title="Rating Distribution",
            labels={"count": "Titles", "label": "Rating"},
            template=theme,
        )

        duration_df = filtered[filtered["duration_int"].notna()]
        duration_fig = px.histogram(
            duration_df,
            x="duration_int",
            color="type",
            nbins=30,
            title="Movie Duration and TV Show Seasons",
            labels={"duration_int": "Duration (min / seasons)", "count": "Titles"},
            template=theme,
            barmode="overlay",
            opacity=0.8,
        )

        col1.plotly_chart(rating_fig, use_container_width=True)
        col2.plotly_chart(duration_fig, use_container_width=True)

    with st.container():
        col1, col2 = st.columns([1, 1])
        added_data = (
            filtered[filtered["added_year"] > 0]
            .groupby("added_year")
            .size()
            .reset_index(name="titles_added")
            .sort_values("added_year")
        )
        added_fig = px.line(
            added_data,
            x="added_year",
            y="titles_added",
            markers=True,
            title="Titles Added to Netflix Per Year",
            labels={"added_year": "Year Added", "titles_added": "Titles Added"},
            template=theme,
        )

        director_df = (
            pd.Series([director for director in filtered["director"].str.split(",").explode().str.strip() if director])
            .value_counts()
            .head(15)
            .reset_index()
            .rename(columns={"index": "director", 0: "count"})
        )
        director_fig = px.bar(
            director_df,
            x="count",
            y="director",
            orientation="h",
            title="Top Directors by Title Count",
            labels={"count": "Titles", "director": "Director"},
            template=theme,
        )

        col1.plotly_chart(added_fig, use_container_width=True)
        col2.plotly_chart(director_fig, use_container_width=True)

    with st.container():
        col1, col2 = st.columns([1, 1])
        cast_flat = [actor for actors in filtered["cast_list"] for actor in actors if actor and actor != "Unknown"]
        cast_df = (
            pd.Series(cast_flat)
            .value_counts()
            .head(20)
            .reset_index()
            .rename(columns={"index": "actor", 0: "count"})
        )
        cast_fig = px.bar(
            cast_df,
            x="count",
            y="actor",
            orientation="h",
            title="Top Cast Members",
            labels={"count": "Credits", "actor": "Actor"},
            template=theme,
        )

        keyword_df = extract_keyword_counts(filtered["description"], top_n=20)
        keyword_fig = px.bar(
            keyword_df,
            x="count",
            y="keyword",
            orientation="h",
            title="Common Keywords in Descriptions",
            labels={"count": "Frequency", "keyword": "Keyword"},
            template=theme,
        )

        col1.plotly_chart(cast_fig, use_container_width=True)
        col2.plotly_chart(keyword_fig, use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"**Average parsed duration:** {average_duration_label}  "
        f"**Filtered titles:** {total_titles:,}  "
        f"**Search text:** `{search_text}`"
    )


if __name__ == "__main__":
    build_dashboard()
