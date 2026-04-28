import pandas as pd
import os
import re
import matplotlib
import numpy as np
import seaborn as sns
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 933120000

from scipy.stats import ttest_ind
import itertools
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import glob
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class GRatioCalc:
    def __init__(self, folder_path, parameters, staining):
        self.folder_path = folder_path
        self.parameters = parameters
        self.animal_data = {}
        self.staining = staining
        self.groups = ''
        self.conversion = ''
        self.reverse_conversion_dict = ''
        self.pp = ''
        self.palette = {"Control 1x": "green", "Control 3x": "green",
                        "Control 5x - 1": "green", "Control 5x - 3": "green", "Control 5x - 6": "green",

                        "Linear 1x": "red", "Linear 3x": "red",
                        "Linear 5x - 1": "red", "Linear 5x - 3": "red", "Linear 5x - 6": "red",

                        "Macro 1x": "blue", "Macro 3x": "blue",
                        "Macro 5x - 1": "blue", "Macro 5x - 3": "blue", "Macro 5x - 6": "blue",

                        "Tm1C homozygous control": "green",
                        "Gfpt1 KO - galactose treated": "red",
                        "Gfpt1 KO - untreated": "blue",
                        'Control': 'green',
                        'GFPT1 + galactose': 'red',
                        'GFPT1': 'blue'}

    def set_staining(self):
        if self.staining == 'TB':
            self.folder_path = Path(self.folder_path).joinpath('tb')

            # Define the groups
            self.groups = {'Control 1x': ['F11', 'F12', 'F13', 'F14', 'F15', 'F26', 'F27', 'F28', 'F29', 'F30'],
                           'Control 3x': ['F41', 'F42', 'F43', 'F44', 'F45', 'F56', 'F57', 'F58', 'F59', 'F60'],
                           'Linear 1x': ['F21', 'F22', 'F23', 'F24', 'F25', 'F36', 'F37', 'F38', 'F39', 'F40'],
                           'Linear 3x': ['F51', 'F52', 'F53', 'F54', 'F55', 'F66', 'F67', 'F68', 'F69', 'F70'],
                           'Macro 1x': ['F16', 'F17', 'F18', 'F19', 'F20', 'F31', 'F32', 'F34', 'F35'],
                           'Macro 3x': ['F46', 'F47', 'F48', 'F61', 'F62', 'F63', 'F64', 'F65'],
                           'Control 5x - 1': ['A31', 'A32', 'A33', 'A34', 'A35',
                                              'A46', 'A47', 'A48', 'A49', 'A50'],
                           'Control 5x - 6': ['A61', 'A62', 'A63', 'A64',
                                              'A72', 'A73', 'A74', 'A75'],
                           'Control 5x - 3': ['A85', 'A86', 'A87', 'A88', 'A89',
                                              'A100', 'A101', 'A102', 'A103', 'A104'],
                           'Macro 5x - 1': ['A36', 'A37', 'A38', 'A39', 'A40',
                                            'A51', 'A52', 'A53', 'A54', 'A55'],
                           'Macro 5x - 6': ['A65', 'A66', 'A67', 'A68',
                                            'A76', 'A77', 'A78', 'A79', 'A80'],
                           'Macro 5x - 3': ['A90', 'A91', 'A92', 'A93', 'A94',
                                            'A105', 'A106', 'A107', 'A108', 'A109'],
                           'Linear 5x - 1': ['A41', 'A42', 'A43', 'A44', 'A45',
                                             'A56', 'A57', 'A58', 'A59', 'A60'],
                           'Linear 5x - 6': ['A69', 'A70', 'A71',
                                             'A81', 'A82', 'A83', 'A84'],
                           'Linear 5x - 3': ['A95', 'A96', 'A97', 'A98', 'A99',
                                             'A110', 'A111', 'A112', 'A113', 'A114']
                           }
            # Missing 49, 50, 33
            self.conversion = {'F11': '2023-12-04 17.12.46', 'F12': '2023-12-04 17.13.46', 'F13': '2023-12-04 17.14.58',
                               'F14': '2024-02-20 18.55.01', 'F15': '2024-02-20 18.54.14', 'F16': '2023-12-04 17.15.47',
                               'F17': '2023-12-04 17.16.28', 'F18': '2023-12-04 17.16.56', 'F19': '2024-02-20 18.53.09',
                               'F20': '2024-02-20 18.51.52',
                               'F21': '2023-12-04 17.17.37', 'F22': '2023-12-04 17.18.30', 'F23': '2023-12-04 17.19.28',
                               'F24': '2024-02-20 18.50.44', 'F25': '2024-02-20 19.07.23', 'F26': '2024-02-20 19.07.59',
                               'F27': '2024-02-20 19.08.44', 'F28': '2024-02-20 19.09.44', 'F29': '2024-02-20 19.10.33',
                               'F30': '2024-02-20 19.11.20', 'F31': '2024-02-20 19.12.15', 'F32': '2024-02-20 19.13.17',
                               'F34': '2024-02-20 19.02.19', 'F35': '2024-02-20 19.03.08', 'F36': '2024-02-20 19.03.47',
                               'F37': '2024-02-20 19.04.25', 'F38': '2024-02-20 19.05.05', 'F39': '2024-02-20 19.05.47',
                               'F40': '2024-02-20 19.06.36', 'F41': '2023-12-04 17.20.27', 'F42': '2023-12-04 17.21.10',
                               'F43': '2023-12-04 17.22.01', 'F44': '2024-02-20 18.55.39', 'F45': '2024-02-20 18.56.21',
                               'F46': '2023-12-04 17.22.50', 'F47': '2023-12-04 17.23.29', 'F48': '2023-12-04 17.24.13',
                               'F51': '2023-12-04 17.25.04', 'F52': '2023-12-04 17.25.50', 'F53': '2023-12-04 17.26.39',
                               'F54': '2024-02-20 18.57.27', 'F55': '2024-02-20 18.58.28', 'F56': '2024-02-20 18.59.26',
                               'F57': '2024-02-20 19.00.40', 'F58': '2024-02-20 19.01.35', 'F59': '2024-01-09 15.46.36',
                               'F60': '2024-01-09 15.38.41', 'F61': '2024-01-09 15.39.36', 'F62': '2024-01-09 15.40.21',
                               'F63': '2024-01-09 15.41.01', 'F64': '2024-01-09 15.41.35', 'F65': '2024-01-09 15.42.10',
                               'F66': '2024-01-09 15.42.46', 'F67': '2024-01-09 15.43.20', 'F68': '2024-01-09 15.44.03',
                               'F69': '2024-01-09 15.44.42', 'F70': '2024-01-09 15.45.27', 'A31': '2025-01-25 12.00.57',

                               'A32': '2025-01-25 12.02.12', 'A33': '2025-01-25 12.03.24', 'A36': '2025-01-25 12.06.11',
                               'A37': '2025-01-25 12.07.45', 'A38': '2025-01-25 12.08.51', 'A40': '2025-01-25 12.11.22',
                               'A41': '2025-01-25 12.11.48', 'A42': '2025-01-25 12.13.19', 'A43': '2025-01-25 12.14.35',
                               'A44': '2025-01-25 12.15.02', 'A45': '2025-01-25 12.16.45', 'A46': '2025-01-25 12.17.41',
                               'A47': '2025-01-25 12.18.58', 'A48': '2025-01-25 12.20.29', 'A49': '2025-01-25 12.21.46',
                               'A71': '2025-01-25 12.24.10', 'A73': '2025-01-25 12.25.56', 'A74': '2025-01-25 12.26.55',
                               'A76': '2025-01-25 12.28.13', 'A77': '2025-01-25 12.29.26', 'A78': '2025-01-25 12.30.55',
                               'A79': '2025-01-25 12.32.21', 'A80': '2025-01-25 12.33.44', 'A82': '2025-01-25 12.34.51',
                               'A84': '2025-01-25 12.35.53', 'A85': '2025-01-25 12.36.34', 'A86': '2025-01-25 12.37.34',
                               'A88': '2025-01-25 12.38.51', 'A89': '2025-01-25 12.40.20', 'A90': '2025-01-25 12.41.31',
                               'A92': '2025-01-25 12.42.49', 'A93': '2025-01-25 12.43.38', 'A94': '2025-01-25 12.44.56',
                               'A95': '2025-01-25 12.46.17', 'A96': '2025-01-25 12.47.28', 'A97': '2025-01-25 12.48.39',
                               'A98': '2025-01-25 12.49.54', 'A99': '2025-01-25 12.51.02', 'A100': '2025-01-25 12.52.12',
                               'A102': '2025-01-25 12.53.02', 'A103': '2025-01-25 12.54.47', 'A104': '2025-01-25 12.56.38',
                               'A106': '2025-01-25 12.57.56', 'A107': '2025-01-25 12.59.22', 'A108': '2025-01-25 13.00.46',
                               'A110': '2025-01-25 13.02.10', 'A111': '2025-01-25 13.03.45', 'A112': '2025-01-25 13.04.51',
                               'A114': '2025-01-25 13.07.44'}

            self.reverse_conversion_dict = {v: k for k, v in self.conversion.items()}

            for group, mice in self.groups.items():
                self.groups[group] = [
                    self.conversion[mouse]
                    for mouse in mice if mouse in self.conversion
                ]

        else:
            self.folder_path = Path(self.folder_path).joinpath('ppd')
            # Define the groups

            self.groups = {'Tm1C homozygous control': ['AA1M3', 'AE2M2', 'AF8M3', 'AF8M4', 'AU2M3', 'AZ2M2', 'AZ2M3', 'BBIF2', 'BD4F6', 'BD4F7'],
                           'Gfpt1 KO - galactose treated': ['AU1M1', 'AU3M4', 'AU3M6', 'AE5M2', 'AI2M3', 'AIIF1', 'AU5F2', 'AZ2M1'],
                           'Gfpt1 KO - untreated': ['AF8M2', 'AGIM2', 'AU2M1', 'AU2M2', 'AU3M5', 'AU7M1', 'BD4F2', 'BD4F4', 'BD4F5']}

    def build_pdf(self):
        self.folder_path.joinpath('plots').mkdir(exist_ok=True, parents=True)
        self.pp = PdfPages(Path(self.folder_path).joinpath(f'plots/{self.staining}_morphometrics.pdf'))

    def save_pdf(self):
        self.pp.close()

    def add_plot(self):
        self.pp.savefig(plt.gcf(), dpi=400)

    def extract_animal_id(self, filename):
        # Function to extract unique animal identifier from filename
        # name_match = re.search(r'Axon_seg-[^_]*?_([^_]+)_', filename)
        # filename = filename.replace('EM2-24_', '')
        name_match = re.search(r'Axon_seg-([^\s_]*)', filename)

        # if False:
        #     name_match = re.search(r'Axon_seg-(.*?)\[d=', filename)
        #     xcoords = re.search(r',x=(.*?)\,y=', filename).group(1)
        #     ycoords = re.search(r',y=(.*?)\,w=', filename).group(1)
        #
        #     if self.staining == 'TB':
        #         return name_match.group(1).strip(), xcoords, ycoords if name_match else None
        #     elif self.staining == 'PPD':
        #         return name_match.group(1).split(' ')[0], xcoords, ycoords if name_match else None
        #     else:
        #         print('Wrong staining!!')

        return name_match.group(1).split(' ')[0] if name_match else None

    # Function to aggregate data
    def aggregate_data(self):
        morphometrics_dir = self.folder_path.joinpath('g_ratio_datafiles')
        all_data = []

        # Iterate over all CSV files in the directory
        for file in glob.glob(str(morphometrics_dir.joinpath('Axon*.csv'))):
            if 'overview' in file:
                continue

            # Read the CSV file
            df = pd.read_csv(file)

            # Extract the filename without extension as animal_id
            # if False:
            #     animal_id, xcoords, ycoords = self.extract_animal_id(file)
            # else:

            # if not re.search(r"_2.+?_", file):
            #     animal_id = self.extract_animal_id(file)
            # else:
            #     name_match = re.search(r'_([^\s_]*)_combined', file)
            name_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}\.\d{2}\.\d{2}', file)
            animal_id = name_match.group() if name_match else None

            try:
                group = next(group for group in self.groups if animal_id in self.groups[group])
            except StopIteration:
                continue  # Replace with your else condition

            # Add the animal_id as a new column
            df['Animal'] = animal_id
            df['Group'] = group
            # if False:
            #     df.loc[df['Animal'] == animal_id, 'xcoords'] = xcoords
            #     df.loc[df['Animal'] == animal_id, 'ycoords'] = ycoords

            # Append to the list
            all_data.append(df)

        # Concatenate all dataframes
        self.animal_data = pd.concat(all_data, ignore_index=True)

        if self.staining == 'TB':
            # Convert 'date' names back to mouse names
            self.animal_data['Animal'] = self.animal_data['Animal'].map(
                self.reverse_conversion_dict).fillna(self.animal_data['Animal'])

        self.animal_data = self.animal_data.sort_values('Animal')

        return self.animal_data

    # Function to create grouped subplots
    def create_stat_plots(self, parameter):
        for nerves_included in ['All']:
        # for nerves_included in ['All', 'Small', 'Large']:
            if nerves_included == 'All':
                size_subset = self.animal_data
            elif nerves_included == 'Small':
                size_subset = self.animal_data[self.animal_data['Axon_Diameter'] < self.animal_data['Axon_Diameter'][self.animal_data['Group'] == 'Gfpt1 KO - galactose treated'].median()]
            elif nerves_included == 'Large':
                size_subset = self.animal_data[self.animal_data['Axon_Diameter'] > self.animal_data['Axon_Diameter'][self.animal_data['Group'] == 'Gfpt1 KO - galactose treated'].median()]

            plt.subplots(1, len(self.groups) + 2,
                         figsize=[(len(self.groups) + 2) * 5, 15], sharey=True)

            for i, group in enumerate(self.groups, start=1):
                plt.subplot(1, len(self.groups) + 2, i)
                sns.boxplot(x='Animal', y=parameter,
                            data=size_subset[size_subset['Group'] == group].sort_values('Animal'))
                plt.xticks(rotation=45)

                if parameter != 'G-ratio':
                    unit = ' μm'
                else:
                    unit = ''

                plt.title(f"{list(self.groups.keys())[i-1]}, "
                          f"mean: {np.round(size_subset[size_subset['Group'] == group][parameter].mean(), 2)}{unit}, "
                          f"SD: {np.round(size_subset[size_subset['Group'] == group][parameter].std(), 2)}{unit}")
                plt.xlabel('Animal ID')
                if parameter != 'G-ratio':
                    plt.ylabel(parameter + ' (μm)')
                else:
                    plt.ylabel(parameter)

            plt.tight_layout()

            # Perform STATISTICS
            group_names = size_subset['Group'].unique()

            if self.staining == 'PPD':
                pairs = list(itertools.combinations(range(len(group_names)), 2))
            else:
                # Desired combinations
                desired_combinations = [('Control 1x', 'Control 3x'),
                                        ('Control 1x', 'Linear 1x'),
                                        ('Control 1x', 'Macro 1x'),
                                        ('Control 3x', 'Linear 3x'),
                                        ('Control 3x', 'Macro 3x')]

                desired_combinations = [('Control 5x - 1', 'Linear 5x - 6'),
                                        ('Control 5x - 1', 'Macro 5x - 6'),
                                        ('Linear 5x - 6', 'Macro 5x - 6')]

                # Extracting indices of the group names
                group_indices = {name: index for index, name in enumerate(group_names)}

                # Generating pairs of indices based on the desired combinations
                pairs = [(group_indices[pair[0]], group_indices[pair[1]]) for pair in desired_combinations]

            for idx, selected_group in enumerate(['Nerves', 'Animals'], start=1):
                if selected_group == 'Animals':
                    size_subset = size_subset.groupby(['Animal', 'Group'])[parameter].mean().reset_index(name=parameter)

                plt.subplot(1, len(self.groups) + 2, len(self.groups) + idx)
                print(idx, selected_group)
                sns.boxplot(x='Group', y=parameter, data=size_subset, palette=self.palette)
                plt.xticks(rotation=45)

                if selected_group == 'Animals':
                    plt.title(f'Group comparison per animal (ttest)')
                else:
                    plt.title(f'Group comparison based on all nerves (ttest)')
                plt.xlabel('Group')
                plt.ylabel(parameter)

                # Perform ANOVA
                anova_data = [size_subset[size_subset['Group'] == group][parameter] for group in group_names]
                anova_stat, anova_pval = f_oneway(*anova_data)

                print(f"ANOVA result: Stat={anova_stat}, p-value={anova_pval}, {parameter}, {nerves_included} nerves, "
                      f"for {selected_group}")

                if anova_pval < 0.05:
                    print("Significant differences found, proceeding with Tukey HSD post hoc test.")

                    # Concatenate all groups for post hoc analysis
                    data_for_posthoc = pd.concat(
                        [size_subset[size_subset['Group'] == group][parameter].reset_index(drop=True) for group in
                         group_names], axis=0)
                    groups_for_posthoc = np.concatenate(
                        [[group] * len(size_subset[size_subset['Group'] == group]) for group in group_names])

                    # Perform Tukey HSD
                    tukey_result = pairwise_tukeyhsd(endog=data_for_posthoc, groups=groups_for_posthoc, alpha=0.05)
                    print(tukey_result)

                    # Plotting (optional) - Adjusted for ANOVA/Tukey results visualization
                    # tukey_result.plot_simultaneous(xlabel=parameter, ylabel='Group')
                    # plt.show()
                else:
                    print("No significant differences found among the groups.")

                y_max = size_subset[parameter].max()
                y_step = y_max * 0.1
                for pair in pairs:
                    group1_data = size_subset[size_subset['Group'] == group_names[pair[0]]][parameter]
                    group2_data = size_subset[size_subset['Group'] == group_names[pair[1]]][parameter]
                    stat, p_val = ttest_ind(group1_data, group2_data)

                    # Calculating positions for significance bars
                    x1, x2 = pair
                    y, h, col = y_max + 0.5 * y_step * pairs.index(pair), y_step * 0.15, 'k'
                    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
                    plt.text((x1 + x2) * .5, y + h, f"p={p_val:.3f}", ha='center', va='bottom', color=col)

                if parameter == 'Axon_Diameter':
                    plt.suptitle(f'{parameter} overview for {nerves_included.lower()} nerve fibers',
                                 y=1,
                                 fontweight="bold")
                else:
                    plt.suptitle(f'{parameter} overview for {nerves_included.lower()} nerve fibers (based on axon diameter)',
                                 y=1,
                                 fontweight="bold")
            plt.savefig(self.folder_path.joinpath(f'plots/{self.staining}_{parameter}_{nerves_included}.png'))

            plt.show()
            self.add_plot()

    def create_size_plots(self):
        # First subplot
        ax1 = sns.lmplot(x='Axon_Diameter', y='Myelin_Thickness', hue='Group', data=self.animal_data)
        plt.xlabel('Axon Diameter (μm)')
        plt.ylabel('Myelin Thickness (μm)')
        plt.show()
        self.add_plot()

        # Second subplot
        ax2 = sns.lmplot(x='Axon_Diameter', y='G-ratio', hue='Group', data=self.animal_data)
        plt.xlabel('Axon Diameter (μm)')
        plt.show()
        self.add_plot()

    def create_count_plots(self):
        for nerves_included in ['All']:
        # for nerves_included in ['All', 'Small', 'Large']:
            if nerves_included == 'All':
                size_subset = self.animal_data
            elif nerves_included == 'Small':
                size_subset = self.animal_data[self.animal_data['Axon_Diameter'] < self.animal_data['Axon_Diameter'].median()]
            elif nerves_included == 'Large':
                size_subset = self.animal_data[self.animal_data['Axon_Diameter'] > self.animal_data['Axon_Diameter'].median()]

            # Determine the number of groups for subplot dimensions
            num_groups = len(self.groups)

            # Define the figure with dynamic sizing based on the number of groups plus one for the overlay plot
            plt.figure(figsize=[5 * (num_groups + 1), 5])

            max_ylim = 0  # Variable to store the maximum y-axis limit

            # Loop through each group in self.groups
            for i, (group_name, animals) in enumerate(self.groups.items()):
                ax = plt.subplot(1, num_groups + 1, i + 1)

                # Calculate and sort the counts for each animal
                counts = []
                for animal in animals:
                    if self.staining == 'TB':
                        # not with A animals
                        # continue
                        # only with F animals:
                        animal_data = size_subset[size_subset['Animal'] == self.reverse_conversion_dict[animal]]
                        animal_selected = self.reverse_conversion_dict[animal]
                    else:
                        animal_data = size_subset[size_subset['Animal'] == animal]
                        animal_selected = animal
                    count = len(animal_data['Axon_Diameter'])
                    counts.append((count, animal_selected, animal_data['Axon_Diameter']))
                counts.sort(reverse=True)  # Sort so that the animal with the most counts is first

                # Plot histogram for each animal in sorted order
                for count, animal, axon_diameters in counts:
                    ax.hist(axon_diameters, bins=50, alpha=0.8, label=animal)

                plt.title(group_name)
                plt.xlabel('Axon Diameter (μm)')
                plt.ylabel('Number of counted axons' if i == 0 else '')  # Only add y-label to the first subplot

                # Update the maximum y-axis limit
                max_ylim = max(max_ylim, ax.get_ylim()[1])

                plt.legend()
                plt.tight_layout()

            # Fourth subplot for the overlay of group histograms
            ax_overlay = plt.subplot(1, num_groups + 1, num_groups + 1)
            # for i, group_name in enumerate(self.groups.keys()):

            # group_data = size_subset[size_subset['Group'] == group_name]
            # ax_overlay.hist(group_data['Axon_Diameter'], bins=50,
            #                 label=self.palette.keys(), color=self.palette.values())
            #
            # sns.histplot(size_subset.melt(), x='value', hue='variable',
            #              multiple='dodge', shrink=.75, bins=50)
            #
            # plt.figure(figsize=(10, 6))

            sns.histplot(data=size_subset, x='Axon_Diameter', hue='Group', fill=False,
                         bins=50, palette=self.palette, legend=self.palette.keys(), ax=ax_overlay)

            plt.title('Overlay of Groups')
            plt.xlabel('Axon Diameter (μm)')
            plt.ylabel('Number of counted axons')
            # plt.legend()
            plt.tight_layout()

            # Adjust the y-axis range of the first three plots
            for j in range(1, num_groups + 1):
                plt.subplot(1, num_groups + 1, j).set_ylim(0, max_ylim)

            # Save and show the plot
            plt.suptitle(f'Counts for {nerves_included.lower()} nerve fibers',
                         y=1,
                         fontweight="bold")
            plt.savefig(self.folder_path.joinpath(f'plots/{self.staining}_group_histograms_{nerves_included}.png'))
            plt.show()
            self.add_plot()

    def add_examples(self, number=0):
        n = 0

        for main_group in self.groups:

            for subdir, _, files in os.walk(self.folder_path.joinpath('exported_images')):
                if n == number:
                    break

                if 'overview' in subdir:
                    continue

                images = [f for f in files if f.endswith(('im.jpg', 'seg.jpg', 'seg_selected.jpg'))]
                images = [image for image in images if not image.startswith('.')]
                images.sort()  # Sort the images for consistent ordering

                try:
                    # ppd?
                    # animal_id = images[0].split(' ')[0]
                    # tb?
                    # animal_id = images[0].split('_')[1]
                    # A animals
                    # name_match = re.search(r'_([^\s_]*)_combined', images[0])
                    # animal_id = name_match.group(1).split(' ')[0] if name_match else None
                    # mix
                    file_id = re.search(r'\d{4}-\d{2}-\d{2} \d{2}\.\d{2}\.\d{2}', images[0])
                    animal_id = file_id.group()

                    if not animal_id.startswith(f'2025'):
                        continue

                except IndexError:
                    print('Index error', files)
                    continue

                if animal_id == 'GFPM':
                    continue

                try:
                    group = next(group for group in self.groups if animal_id in self.groups[group])
                except StopIteration:
                    continue  # Replace with your else condition

                if not group in main_group:
                    continue

                # If there are three images as expected
                if len(images) == 3:
                    # Check dimensions before proceeding
                    for img_name in images:
                        img_path = os.path.join(subdir, img_name)
                        with PIL.Image.open(img_path) as img:
                            width, height = img.size
                        if width > 20000 or height > 20000:
                            break  # Skip processing if any image is too large
                    else:  # Only execute if all images are within the size limit
                        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns
                        titles = ['Image', 'Raw segmentation', 'Selected segmentation']

                        # Plot each image
                        for idx, img_name in enumerate(images):
                            img_path = os.path.join(subdir, img_name)
                            img = plt.imread(img_path)
                            axs[idx].imshow(img)
                            axs[idx].set_title(animal_id + ', ' + group + ', ' + titles[idx] + ' ' + str(n))
                            axs[idx].axis('off')

                        plt.show()
                        self.add_plot()
                        n += 1

# Specify the parameters and folder path here
# root_dir = '/Volumes/T5_Bart 1/raw_data_TIS/g-ratio/Service/20250210/analysis'
root_dir = '/Volumes/T7_Blue/TIS/g-ratio/Service/20260227/analysis'
parameters = ['G-ratio', 'Axon_Diameter', 'Myelin_Thickness']  # Add more parameters as needed
tb_or_ppd = 'TB'

g_ratio_calculator = GRatioCalc(root_dir, parameters, tb_or_ppd)
g_ratio_calculator.set_staining()
animal_dat = g_ratio_calculator.aggregate_data()
g_ratio_calculator.build_pdf()


for parameter in parameters:
    g_ratio_calculator.create_stat_plots(parameter)

g_ratio_calculator.create_count_plots()
g_ratio_calculator.create_size_plots()

g_ratio_calculator.add_examples(0)
g_ratio_calculator.save_pdf()
