# Dataset provenance and limits

The repository already contains `Mobiles Dataset (2025).csv` and `Mobiles_Cleaned.csv`. The cleaned file contains 830 rows before the application's cleaning step. The audited repository does not establish the original download URL, creator, license, collection procedure or cleaning notebook. No new license or redistribution permission is inferred.

Required source columns include brand/model, RAM, front/rear camera, processor, battery, screen size, US launch price and launch year. Optional numeric columns are preferred when present. Tablet-like names are excluded; rows missing required numeric features are excluded. Cleaning therefore changes the usable count.

Prices are launch USD values. Processor scores are hand-written segment estimates. Camera megapixels are not image-quality measurements. Membership functions and rule scores have no labeled preference study behind them. The project is a rules-based demonstration, not a live shopping recommendation service.

Owner action: supply the original dataset page and applicable terms, document modifications and identify whether either CSV can be redistributed. Until then, do not mirror it into another portfolio repository or assert an open-data license.
