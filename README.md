# CrossGTP-submit
Open source code and data for CrossGTP in submission. 

## Data
Please download data from [here](https://drive.google.com/file/d/1knOIvI6BUpp-A2AZQY-8xh07qTf7t5tC/view?usp=sharing) and unzip them. 

The zip file contain data for cities New York, Chicago, Washington DC and Boston. For each city `X=NY, CHI, DC, BOS`, there are the following data files:
- `BikeX_pickup.npy` and `BikeX_dropoff.npy`. 
- `TaxiX_pickup.npy` and `TaxiX_dropoff.npy`. 
- `X_poi.npy`
- `X_roads.npy`
- `TaxiX_ODPairs`

## Code

After settling down some path and dependency issues, you can run the domain adaptive region embeddings via: 

`python run_mvgcn_da.py --scity X --tcity Y`

You can also personalize parameters. run `python run_mvgcn_da.py --help` for detailed parameter list.

After training the region embeddings, you can run CrossGTP via: 

`python run_crossgtp.py --scity X --tcity Y --source_model MODELPATH --emb_name EMBPATH`

where `MODELPATH` denotes a path to load the pretrained source model, and `EMBPATH` denote the path to the learned embeddings.

Again, you may need to settle some minor path issues. You can also set personalized parameters (See `python run_crossgtp.py --help` for details). 
