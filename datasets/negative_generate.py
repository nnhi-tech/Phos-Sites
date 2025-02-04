import pandas as pd
import random
import re

def negative_position_generate(file_name):
    # Load data, remove duplicates
    # acc: accession number, sequence: protein sequence, position: position of the site, code: amino acid
    positive_position = pd.read_csv(file_name, usecols=['acc', 'sequence', 'position', 'code']) 
    positive_position = positive_position.drop_duplicates()
        
    # Group 'S' and 'T' into one category
    positive_position['code'] = positive_position['code'].replace({'S': 'ST', 'T': 'ST'})
        
    # Add 'label' column with value 1
    positive_position['label'] = 1       
    
    # Count the number of 'S/T' and 'Y' in 'code' column
    num_ST = positive_position['code'].value_counts().get('ST', 0)
    num_Y = positive_position['code'].value_counts().get('Y', 0)
    
    # Generate negative_position dataframe  
    negative_position = positive_position.groupby('acc').agg({
        'position': set,
        'code': 'first',
        'sequence': 'first'
    }).reset_index()
    negative_position.rename(columns={'position': 'positive_position'}, inplace=True)
       
    # Select n random positions from negative_positions
    def random_choice_in_mixture(negative_position):
        # Add acc before each element in positive_position list
        positive_position['position'] = (positive_position['acc'] + ' ' + positive_position['position'].astype(str))
        
        # Get all positions of 'ST' and 'Y' in each sequence
        def get_all_positions(acc, sequence, code):
            if code == 'ST':
                return {acc + ' ' + str(m.start() + 1) for m in re.finditer('S|T', sequence)}
            else:
                return {acc + ' ' + str(m.start() + 1) for m in re.finditer(code, sequence)}

        # Get negative_position
        negative_position['negative_position'] = negative_position.apply(
            lambda row: list(get_all_positions(row['acc'], row['sequence'], row['code']) - row['positive_position']), axis=1)
        
        # Save negative_positions in list
        ST_position = negative_position[negative_position['code'] == 'ST']['negative_position'].sum()
        Y_position = negative_position[negative_position['code'] == 'Y']['negative_position'].sum()
        
        # Randomly choose num_ST and num_Y elements from ST_position and Y_position
        if len(ST_position) >= num_ST:
            ST_position = random.sample(ST_position, num_ST)

        if len(Y_position) >= num_Y:
            Y_position = random.sample(Y_position, num_Y)
        
        # Create DataFrame for ST_position and Y_position
        ST_df = pd.DataFrame(ST_position, columns=['acc'])
        ST_df['code'] = 'ST'
        
        Y_df = pd.DataFrame(Y_position, columns=['acc'])
        Y_df['code'] = 'Y'

        # Concatenate ST_df and Y_df to negative_position
        negative_position = pd.concat([ST_df, Y_df], ignore_index=True)           
        negative_position[['acc', 'position']] = negative_position['acc'].str.split(' ', expand=True)
        positive_position[['acc', 'position']] = positive_position['position'].str.split(' ', expand=True)
        acc_to_sequence = positive_position.set_index('acc')['sequence'].to_dict()
        negative_position['sequence'] = negative_position['acc'].map(acc_to_sequence)
        return negative_position

    negative_position = random_choice_in_mixture(negative_position)
    
    # Add 'label' column with value 0
    negative_position['label'] = 0

    # Concatenate positive_position and negative_position
    position = pd.concat([positive_position, negative_position])
    return position

if __name__ == '__main__':
    position = negative_position_generate('datasets/ELM.csv')

    # Save position to raw_data.csv
    position.to_csv('datasets/raw_data.csv', index=False)