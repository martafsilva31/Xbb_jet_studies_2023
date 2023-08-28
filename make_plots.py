import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import uproot
import pickle
import os
import math



#Function that returns all dataframes in each folder 
def read_dataframes(mu, vr_type, output_directory):
    df_2s = pd.read_pickle(os.path.join(output_directory, f"{vr_type}_df_2s.pkl"))
    df_2bs = pd.read_pickle(os.path.join(output_directory, f"{vr_type}_df_2bs.pkl"))
    df = pd.read_pickle(os.path.join(output_directory, f"{vr_type}_df.pkl"))
    fatjet_df = pd.read_pickle(os.path.join(output_directory, f"{vr_type}_fatjet_df.pkl"))
    return df_2s, df_2bs, df, fatjet_df

#Function that returns all possible scenarios: SHC as altVR, Dan as altVR and Dan as VRJetRho30
def process_scenario(directory,functions, mu_values, vr_types):
    pt_bins = np.arange(250, 1000, 50)

    all_data = {}  # Dictionary to store data for the scenario
    for function in functions:
        all_data[function] = {} # Initialize the inner dictionary for each function
        for mu in mu_values:
            output_directory = os.path.join(directory, f"{function}_{mu}mu")
            mu_data = {}  # Dictionary to store data for each mu value

            if function == "SHC":
                vr_type = "AltVRJet"  # Only process "AltVRJet" for the "SHC" case
                df_2s, df_2bs, df, fatjet_df = read_dataframes(mu, vr_type, output_directory)
                label = "Shrinking Cone Alternative"  
                title = f"$\mu = {mu}$"
                mu_data[vr_type] = {
                    'df_2s': df_2s,
                    'df_2bs': df_2bs,
                    'df': df,
                    'fatjet_df': fatjet_df,
                    'label': label,
                    'title': title
                }
            else:
                for vr_type in vr_types:
                    df_2s, df_2bs, df, fatjet_df = read_dataframes(mu, vr_type, output_directory)
                    label = "AltVR (p=0.1)" if vr_type == "AltVRJet" else r"VR Jet ($\rho = 30$)"
                    title = f"$\mu = {mu}$"
                    mu_data[vr_type] = {
                        'df_2s': df_2s,
                        'df_2bs': df_2bs,
                        'df': df,
                        'fatjet_df': fatjet_df,
                        'label': label,
                        'title': title
                    }

            all_data[function][mu] = mu_data

    return all_data


def clopper_pearson_interval(k, n, confidence):
    # k: Number of successes (double-b events)
    # n: Total number of trials (total events)
    # confidence: Confidence level for the interval (e.g., 0.6827 for 68.27%)
    
    alpha = (1 - confidence) / 2
    lower_bound = 0 if k == 0 else scipy.stats.beta.ppf(alpha, k, n - k + 1)
    upper_bound = 1 if k == n else scipy.stats.beta.ppf(1 - alpha, k + 1, n - k)
    
    return lower_bound, upper_bound

# Function to calculate the b matching efficiency and the corresponding clopper pearson interval for the uncertanties
def b_matching_efficiency(pt_bins,df_2s, df_2bs ,confidence=0.95):

    # Create the total histogram
    total_hist, _ = np.histogram(df_2s["FatJet.TrimmedP4[5].PT"], bins=pt_bins)

    # Create the double b-matched histogram
    double_b_hist, _ = np.histogram(df_2bs["FatJet.TrimmedP4[5].PT"], bins=pt_bins)

    # Calculate the efficiency as the ratio of double b-tagged over total
    efficiency = double_b_hist / total_hist
    
    # Calculate the lower and upper bounds using the clopper_pearson_interval function
    lower_bound = np.zeros(len(double_b_hist))
    upper_bound = np.zeros(len(double_b_hist))

    for i, (k, n) in enumerate(zip(double_b_hist, total_hist)):
        lower_bound[i], upper_bound[i] = clopper_pearson_interval(k, n, confidence)
        
    lower_bound = efficiency - lower_bound
    upper_bound = upper_bound - efficiency
    
    return efficiency, lower_bound, upper_bound

#Plot the b matching efficiency for the 3 scenarios in one plot; 3 plots each for mu=0,50,100

def plot_efficiency_all(all_data, output_dir):
    vr_types = ["AltVRJet", "VRJetRho30"]
    mu_values = [0, 50, 100]
    pt_bins = np.arange(250, 1000, 50)
    confidence = 0.95  # Confidence level for error bars
    
    legend_format = ['b.', "rs", 'g^']

    for mu in mu_values:
        i = 0
        plt.figure()  
        plt.title(f"$\mu$ = {mu}")
        plt.xlabel('Higgs Jet $p_T$ (GeV)')
        plt.ylabel('Double Subjet B-matching Efficiency')
        plt.ylim(0.1, 1.1)
        
        for function, mu_data_dict in all_data.items():
            if mu not in mu_data_dict:
                continue

            data = mu_data_dict[mu]

            for vr_type in vr_types:
                if vr_type not in data:
                    continue

                df_2s = data[vr_type]['df_2s']
                df_2bs = data[vr_type]['df_2bs']

                efficiency, lower_bound, upper_bound = b_matching_efficiency(pt_bins, df_2s, df_2bs, confidence)

                label = data[vr_type]['label']
                title = data[vr_type]['title']

                pt_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
                bin_widths = pt_bins[1:] - pt_bins[:-1]
                
                plt.errorbar(pt_centers, efficiency, xerr=bin_widths/2, yerr=[lower_bound, upper_bound],
                             fmt=legend_format[i], label=label, capsize=3, elinewidth=1)
                i += 1

        plt.legend(loc='lower left')
        plt.ylim(0,1.1)

        # Save the plot in a pdf
        output_file = os.path.join(output_dir, f"efficiency_plot_mu_{mu}.pdf")
        plt.savefig(output_file)

        plt.close()
    
#Plot the efficiency as a function of mu for each scenario     
def plot_efficiency_mus(all_data, output_dir):
    vr_types = ["AltVRJet", "VRJetRho30"]
    mu_values = [0, 50, 100]
    pt_bins = np.arange(250, 1000, 50)
    confidence = 0.95  # Confidence level for error bars

    legend_format = ['b.', "gs", 'r^']

    for function in all_data:
        if function == "SHC":
            name = "Shrinking Cone Alternative"
            vr_type = "AltVRJet"
            save = "SHC"

            
            for i, mu in enumerate(mu_values):
                df_2s = all_data[function][mu][vr_type]['df_2s']
                df_2bs = all_data[function][mu][vr_type]['df_2bs']

                efficiency, lower_bound, upper_bound = b_matching_efficiency(pt_bins, df_2s, df_2bs, confidence)

                pt_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
                bin_widths = pt_bins[1:] - pt_bins[:-1]

                plt.errorbar(pt_centers, efficiency, xerr=bin_widths/2, yerr=[lower_bound, upper_bound],
                            fmt=legend_format[i], label=f"$\mu$ = {mu}", capsize=3, elinewidth=1)

            plt.title(f"{name}")
            plt.xlabel('Higgs Jet $p_T$ (GeV)')
            plt.ylabel('Double Subjet B-matching Efficiency')
            plt.ylim(0,1.1)
            plt.legend()
            output_file = os.path.join(output_dir, f"efficiency_plot_{save}.pdf")
            plt.savefig(output_file)
            plt.close()
        
        else:
            for vr_type in vr_types:
                if vr_type == "AltVRJet":
                    name = "AltVR (p=0.1)"
                    save = "AltVR"
                else:
                    name = r"VR Jet ($\rho = 30$)"
                    save = "VRJetRho30"
                    
                for i, mu in enumerate(mu_values):
                    df_2s = all_data[function][mu][vr_type]['df_2s']
                    df_2bs = all_data[function][mu][vr_type]['df_2bs']

                    efficiency, lower_bound, upper_bound = b_matching_efficiency(pt_bins, df_2s, df_2bs, confidence)

                    pt_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
                    bin_widths = pt_bins[1:] - pt_bins[:-1]

                    plt.errorbar(pt_centers, efficiency, xerr=bin_widths/2, yerr=[lower_bound, upper_bound],
                                fmt=legend_format[i], label=f"$\mu$ = {mu}", capsize=3, elinewidth=1)

                plt.title(f"{name}")
                plt.xlabel('Higgs Jet $p_T$ (GeV)')
                plt.ylabel('Double Subjet B-matching Efficiency')
                plt.ylim(0,1.1)
                plt.legend()
                output_file = os.path.join(output_dir, f"efficiency_plot_{save}.pdf")
                plt.savefig(output_file)
                plt.close()

#Plot the Higgs pT histogram used when calculating the efficiency
def plot_pT_Higgs(all_data, output_dir):
    vr_types = ["AltVRJet", "VRJetRho30"]
    mu_values = [0, 50, 100]
    pt_bins = np.linspace(250, 800, 71)
    
    
    for function in all_data:
        if function == "SHC":
            vr_type = "AltVRJet"
            save = "SHC"
            name = "SHrinking Cone Alternative"
        
            for  mu in mu_values:
                
                df_2s = all_data[function][mu][vr_type]['df_2s']
                df_2bs = all_data[function][mu][vr_type]['df_2bs']
                

                plt.hist(df_2s["FatJet.TrimmedP4[5].PT"], bins=pt_bins, density=False, alpha=0.7, histtype='step', label="Higgs Jet",
                        linewidth=1, color="b")

                plt.hist(df_2bs["FatJet.TrimmedP4[5].PT"], bins=pt_bins, density=False, alpha=0.7, histtype='step', label="Higgs Jet with Subjets B-matched",
                        linewidth=1, color="g")

                
                plt.title(f"{name} - $\mu$ = {mu}")
                plt.xlabel("pT")
                plt.legend()
                output_file = os.path.join(output_dir, f"pT_Higgs_{save}_{mu}.pdf")
                plt.savefig(output_file)
                plt.close()
                
        else:
            for vr_type in vr_types:
                if vr_type == "AltVRJet":
                    name = "AltVR (p=0.1)"
                    save = "AltVR"
                else:
                    name = r"VR Jet ($\rho = 30$)"
                    save = "VRJetRho30"
                    
                for mu in mu_values:
                    
                    df_2s = all_data[function][mu][vr_type]['df_2s']
                    df_2bs = all_data[function][mu][vr_type]['df_2bs']
                    

                    plt.hist(df_2s["FatJet.TrimmedP4[5].PT"], bins=pt_bins, density=False, alpha=0.7, histtype='step', label="Higgs Jet",
                            linewidth=1, color="b")

                    plt.hist(df_2bs["FatJet.TrimmedP4[5].PT"], bins=pt_bins, density=False, alpha=0.7, histtype='step', label="Higgs Jet with Subjets B-matched",
                            linewidth=1, color="g")

                    
                    plt.title(f"{name} - $\mu$ = {mu}")
                    plt.xlabel("pT")
                    plt.legend()
                    output_file = os.path.join(output_dir, f"pT_Higgs_{save}_{mu}.pdf")
                    plt.savefig(output_file)
                    plt.close()      

#Functions to plot the Fatjet Trimmed Mass 
def plot_fatjet_trimmed_mass(df_list, mus, output_dir):
    rang = np.linspace(0, 200, 50)
    labels = [f"$\mu = {mu}$" for mu in mus]
    colors = ["b", "g", "r"]

    plt.hist([df["FatJet.TrimmedP4[5].Mass"] for df in df_list], bins=rang, density=True, alpha=0.7, histtype='step', label=labels, linewidth=1, color=colors)

    plt.xlabel("Fatjet Trimmed Mass")
    plt.ylabel("Normalized Number of Fatjets")
    plt.legend()
    output_file = os.path.join(output_dir, "Fatjet_trimmed_mass.pdf")
    plt.savefig(output_file)
    plt.close()

def plot_fatjet_trimmed_mass_pT(df_list, mus, output_dir):
    #Receveives fatjet trimmed with a pT>250GeV and eta < 2
    rang = np.linspace(0, 200, 50)
    labels = [f"$\mu = {mu}$" for mu in mus]
    colors = ["b", "g", "r"]
    
    plt.hist([df["FatJet.TrimmedP4[5].Mass"] for df in df_list], bins=rang, density=True, alpha=0.7, histtype='step', label=labels, linewidth=1, color = colors)

    plt.xlabel("Fatjet Trimmed Mass")
    plt.ylabel("Normalized Number of Fatjets")
    plt.legend()
    output_file = os.path.join(output_dir, "Fatjet_trimmed_mass_pT_cut.pdf")
    plt.savefig(output_file)
    plt.close()

 
def plot_fatjet_trimmed_mass_mass(df_list, mus, output_dir):
    rang = np.linspace(74, 151, 50)
    labels = [f"$\mu = {mu}$" for mu in mus]
    colors = ["b", "g", "r"]

    plt.hist([df["FatJet.TrimmedP4[5].Mass"] for df in df_list], bins=rang, density=True, alpha=0.7, histtype='step', label=labels, linewidth=1, color = colors)

    plt.xlabel("Fatjet Trimmed Mass")
    plt.ylabel("Normalized Number of Fatjets")
    plt.legend()
    output_file = os.path.join(output_dir, "Fatjet_trimmed_mass_mass_cut.pdf")
    plt.savefig(output_file)
    plt.close()
    
#Plot the pT of the lead and subleading subjet 

def get_subjet_pt(df_2s):
    # Get the arrays of leading and subleading subjet PT
    subjet_pt_lead = df_2s['leading_subjets.PT'].apply(lambda pt_array: pt_array[0] )
    subjet_pt_sublead =df_2s['leading_subjets.PT'].apply(lambda pt_array: pt_array[1] )
    return subjet_pt_lead, subjet_pt_sublead

def plot_leading_subjet_pt(all_data, output_dir):
    vr_types = ["AltVRJet", "VRJetRho30"]
    mu_values = [0, 50, 100]
    pt_bins = np.linspace(0,300,70)
    colors = ["b", "g", "r"]
    
    for function in all_data:
        if function == "SHC":
            vr_type = "AltVRJet"
            
            plt.figure()
            for i, mu in enumerate(mu_values):
                df_2s = all_data[function][mu][vr_type]['df_2s']
                subjet_pt_lead, subjet_pt_sublead = get_subjet_pt(df_2s)
        
                plt.hist(subjet_pt_lead, bins=pt_bins, density=True, alpha=0.7, histtype='step', label=f"$\mu$ = {mu}",
                         linewidth=1, color=colors[i])
                
            plt.title("Leading VR Subjet pT - Shrinking Cone Alternative")
            plt.xlabel(" $p_T$ [GeV]")
            plt.ylabel("Entries")
            plt.yscale("log")
            plt.xlim(0,300)
            plt.legend()
            output_file = os.path.join(output_dir, f"leading_subjet_pt_SHC.pdf")
            plt.savefig(output_file)
            plt.close()
  
   
        else:
            for vr_type in vr_types:
                if vr_type == "AltVRJet":
                    name = "AltVR (p=0.1)"
                    save = "AltVR"
                else:
                    name = r"VR Jet ($\rho = 30$)"
                    save = "VRJetRho30"
                for i, mu in enumerate(mu_values):
                    df_2s = all_data[function][mu][vr_type]['df_2s']
                    subjet_pt_lead, subjet_pt_sublead = get_subjet_pt(df_2s)
            
                    plt.hist(subjet_pt_lead, bins=pt_bins, density=True, alpha=0.7, histtype='step', label=f"$\mu$ = {mu}",
                            linewidth=1, color=colors[i])
                    
                plt.title(f"Leading VR Subjet pT - {name}")
                plt.xlabel(" $p_T$ [GeV]")
                plt.ylabel("Entries")
                plt.yscale("log")
                plt.xlim(0,300)
                plt.legend()
                output_file = os.path.join(output_dir, f"leading_subjet_pt_{save}.pdf")
                plt.savefig(output_file)
                plt.close()


def plot_subleading_subjet_pt(all_data, output_dir):
    vr_types = ["AltVRJet", "VRJetRho30"]
    mu_values = [0, 50, 100]
    pt_bins = np.linspace(0,300,70)
    colors = ["b", "g", "r"]
    
    for function in all_data:
        if function == "SHC":
            vr_type = "AltVRJet"
            
            plt.figure()
            for i, mu in enumerate(mu_values):
                df_2s = all_data[function][mu][vr_type]['df_2s']
                subjet_pt_lead, subjet_pt_sublead = get_subjet_pt(df_2s)
        
                plt.hist(subjet_pt_sublead, bins=pt_bins, density=True, alpha=0.7, histtype='step', label=f"$\mu$ = {mu}",
                         linewidth=1, color=colors[i])
                
            plt.title("Subleading VR Subjet pT - Shrinking Cone Alternative")
            plt.xlabel(" $p_T$ [GeV]")
            plt.ylabel("Entries")
            plt.yscale("log")
            plt.legend()
            output_file = os.path.join(output_dir, f"subleading_subjet_pt_SHC.pdf")
            plt.savefig(output_file)
            plt.close()
  
   
        else:
            for vr_type in vr_types:
                if vr_type == "AltVRJet":
                    name = "AltVR (p=0.1)"
                    save = "AltVR"
                else:
                    name = r"VR Jet ($\rho = 30$)"
                    save = "VRJetRho30"
                for i, mu in enumerate(mu_values):
                    df_2s = all_data[function][mu][vr_type]['df_2s']
                    subjet_pt_lead, subjet_pt_sublead = get_subjet_pt(df_2s)
            
                    plt.hist(subjet_pt_sublead, bins=pt_bins, density=True, alpha=0.7, histtype='step', label=f"$\mu$ = {mu}",
                            linewidth=1, color=colors[i])
                    
                plt.title(f"Subleading VR Subjet pT - {name}")
                plt.xlabel(" $p_T$ [GeV]")
                plt.ylabel("Entries")
                plt.yscale("log")
                plt.legend()
                output_file = os.path.join(output_dir, f"subleading_subjet_pt_{save}.pdf")
                plt.savefig(output_file)
                plt.close()

#Plot the number of subjets that passed the HIggs dR matching 

def plot_num_subjets_associated_to_higgs(all_data, output_dir):
    vr_types = ["AltVRJet", "VRJetRho30"]
    mu_values = [0, 50, 100]
    subjet_bins = 50  
    colors = ["b", "g", "r"]
    
    for function in all_data:
        if function == "SHC":
            vr_type = "AltVRJet"
            
            plt.figure()
            for i, mu in enumerate(mu_values):
                df = all_data[function][mu][vr_type]['df']
                num_subjets = df["nb_subjets"]
                
                plt.hist(num_subjets, bins=subjet_bins, density=False, alpha=0.7, histtype='step', label=f"$\mu$ = {mu}",
                         linewidth=1, color=colors[i])
                
            plt.title("Number of Subjets Associated with the Higgs - Shrinking Cone Alternative")
            plt.xlabel("Number of subjets associated to the Higgs")
            plt.ylabel("Number of Events")
            plt.legend()
            output_file = os.path.join(output_dir, f"num_subjets_higgs_SHC.pdf")
            plt.savefig(output_file)
            plt.close()
   
        else:
            for vr_type in vr_types:
                if vr_type == "AltVRJet":
                    name = "AltVR (p=0.1)"
                    save = "AltVR"
                else:
                    name = r"VR Jet ($\rho = 30$)"
                    save = "VRJetRho30"
                    
                for i, mu in enumerate(mu_values):
                    df = all_data[function][mu][vr_type]['df']
                    num_subjets = df["nb_subjets"]
            
                    plt.hist(num_subjets, bins=subjet_bins, density=False, alpha=0.7, histtype='step', label=f"$\mu$ = {mu}",
                            linewidth=1, color=colors[i])
                    
                plt.title(f"Number of Subjets Associated with the Higgs - {name}")
                plt.xlabel("Number of subjets associated to the Higgs")
                
                plt.legend()
                output_file = os.path.join(output_dir, f"num_subjets_higgs_{save}.pdf")
                plt.savefig(output_file)
                plt.close()
                

def plot_num_subjets_associated_to_higgs_mus(all_data, output_dir):
    vr_types = ["AltVRJet", "VRJetRho30"]
    mu_values = [0, 50, 100]
    subjet_bins = 50
    
    for mu in mu_values:
        plt.figure()
        for function in all_data:
            if function == "SHC":
                vr_type = "AltVRJet"
                c = "b"
                df = all_data[function][mu][vr_type]['df']
                label = f"SHC - $\mu$ = {mu}"
                num_subjets = df["nb_subjets"]
                plt.hist(num_subjets, bins=subjet_bins, density=False, alpha=0.7, histtype='step', label=label,
                     linewidth=1, color=c)
                
            else:
                for vr_type in vr_types:
                    if vr_type == "AltVRJet":
                        name = "AltVR (p=0.1)"
                        c = "g"
                    else:
                        name = r"VR Jet ($\rho = 30$)"
                        c = "r"
                        
                    df = all_data[function][mu][vr_type]['df']
                    label = f"{name} - $\mu$ = {mu}"
                    
                    num_subjets = df["nb_subjets"]
                    plt.hist(num_subjets, bins=subjet_bins, density=False, alpha=0.7, histtype='step', label=label,
                            linewidth=1, color=c)
            
        plt.title(f" $\mu$ = {mu}")
        plt.xlabel("Number of subjets associated to the Higgs")
        plt.legend()
        output_file = os.path.join(output_dir, f"num_subjets_higgs_mu_{mu}.pdf")
        plt.savefig(output_file)
        plt.close()

#Study the VR collinear subjets 
def overlap(df_2s):
    df_2s['min_radius'] = df_2s[["sj1_radius", "sj2_radius"]].min(axis=1)

    overlap_df = df_2s[np.log(df_2s["dR_between_vr"]/df_2s['min_radius'])<0]
    non_overlap_df = df_2s[np.log(df_2s["dR_between_vr"]/df_2s['min_radius'])>0]
    all = np.log(df_2s["dR_between_vr"]/df_2s['min_radius'])
    
    return overlap_df , non_overlap_df, all


def plot_overlap_hist(all_data, output_dir):
    vr_types = ["AltVRJet", "VRJetRho30"]
    mu_values = [0, 50, 100]
    bins = 71
    colors = ["b", "g", "r"]
    
    for function in all_data:
        if function == "SHC":
            vr_type = "AltVRJet"
            
            plt.figure()
            for i, mu in enumerate(mu_values):
                df_2s = all_data[function][mu][vr_type]['df_2s']
                overlap_df , non_overlap_df, all = overlap(df_2s)
        
                plt.hist(all, bins=bins, density=True, alpha=0.7, histtype='step', label=f"$\mu$ = {mu}",
                         linewidth=1, color=colors[i])
                
            plt.title("Shrinking Cone Alternative")
            plt.xlabel(r" Log$(\Delta R_{jj}/R_{min})$ ")
            plt.axvline(0, color='k', ls ='--')
            plt.yscale("log")
            plt.legend()
            output_file = os.path.join(output_dir, f"overlap_hist_SHC.pdf")
            plt.savefig(output_file)
            plt.close()
  
   
        else:
            for vr_type in vr_types:
                if vr_type == "AltVRJet":
                    name = "AltVR (p=0.1)"
                    save = "AltVR"
                else:
                    name = r"VR Jet ($\rho = 30$)"
                    save = "VRJetRho30"
                for i, mu in enumerate(mu_values):
                    df_2s = all_data[function][mu][vr_type]['df_2s']
                    overlap_df , non_overlap_df, all= overlap(df_2s)
            
                    plt.hist(all, bins=bins, density=True, alpha=0.7, histtype='step', label=f"$\mu$ = {mu}",
                            linewidth=1, color=colors[i])
                    
                plt.title(f"{name}")
                plt.xlabel(r" Log$(\Delta R_{jj}/R_{min})$ ")
                plt.axvline(0, color='k', ls = '--')
                plt.yscale("log")
                plt.legend()
                output_file = os.path.join(output_dir, f"overlap_hist_{save}.pdf")
                plt.savefig(output_file)
                plt.close()
                
# Calculate the fraction of events that are overlapped 
def overlap_percentage(pt_bins, df_2s, confidence=0.95):
    overlap_df , non_overlap_df, all = overlap(df_2s)

    # Create the total histogram
    total_hist, _ = np.histogram(df_2s["FatJet.TrimmedP4[5].PT"], bins=pt_bins)

    # Create the overlapped histogram
    overlap_hist, _ = np.histogram(overlap_df["FatJet.TrimmedP4[5].PT"], bins=pt_bins)

    # Calculate the efficiency as the ratio of double b-tagged over total
    efficiency = overlap_hist / total_hist
    
    # Calculate the lower and upper bounds using the clopper_pearson_interval function
    lower_bound = np.zeros(len(overlap_hist))
    upper_bound = np.zeros(len(overlap_hist))

    for i, (k, n) in enumerate(zip(overlap_hist, total_hist)):
        lower_bound[i], upper_bound[i] = clopper_pearson_interval(k, n, confidence)
        
    lower_bound = efficiency - lower_bound
    upper_bound = upper_bound - efficiency
    
    return efficiency, lower_bound, upper_bound

def plot_overlap_percentage_all(all_data, output_dir):
    vr_types = ["AltVRJet", "VRJetRho30"]
    mu_values = [0, 50, 100]
    pt_bins = np.arange(250, 1000, 50)
    confidence = 0.95  # Confidence level for error bars
    
    legend_format = ['b.', "rs", 'g^']

    for mu in mu_values:
        i = 0
        plt.figure()  
        plt.title(f"$\mu$ = {mu}")
        plt.xlabel('Higgs Jet $p_T$ (GeV)')
        plt.ylabel(r'Fraction of Events with Log$(\Delta R_{jj}/R_{min}) < 0$ ')
        plt.ylim(0.1, 1.1)
        
        for function, mu_data_dict in all_data.items():
            if mu not in mu_data_dict:
                continue

            data = mu_data_dict[mu]

            for vr_type in vr_types:
                if vr_type not in data:
                    continue

                df_2s = data[vr_type]['df_2s']
                

                efficiency, lower_bound, upper_bound = overlap_percentage(pt_bins, df_2s, confidence)

                label = data[vr_type]['label']
                title = data[vr_type]['title']

                pt_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
                bin_widths = pt_bins[1:] - pt_bins[:-1]
                
                plt.errorbar(pt_centers, efficiency, xerr=bin_widths/2, yerr=[lower_bound, upper_bound],
                             fmt=legend_format[i], label=label, capsize=3, elinewidth=1)
                i += 1

        plt.legend(loc='upper left')

        # Save the plot in a pdf
        output_file = os.path.join(output_dir, f"overlap_percentage_mu_{mu}.pdf")
        plt.savefig(output_file)

        plt.close()
        
def plot_overlap_percentage_mus(all_data, output_dir):
    vr_types = ["AltVRJet", "VRJetRho30"]
    mu_values = [0, 50, 100]
    pt_bins = np.arange(250, 1000, 50)
    confidence = 0.95  # Confidence level for error bars

    legend_format = ['b.', "gs", 'r^']

    for function in all_data:
        if function == "SHC":
            name = "Shrinking Cone Alternative"
            vr_type = "AltVRJet"
            save = "SHC"

            
            for i, mu in enumerate(mu_values):
                df_2s = all_data[function][mu][vr_type]['df_2s']
                

                efficiency, lower_bound, upper_bound = overlap_percentage(pt_bins, df_2s, confidence)

                pt_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
                bin_widths = pt_bins[1:] - pt_bins[:-1]

                plt.errorbar(pt_centers, efficiency, xerr=bin_widths/2, yerr=[lower_bound, upper_bound],
                            fmt=legend_format[i], label=f"$\mu$ = {mu}", capsize=3, elinewidth=1)

            plt.title(f"{name}")
            plt.xlabel('Higgs Jet $p_T$ (GeV)')
            plt.ylabel(r'Fraction of Events with Log$(\Delta R_{jj}/R_{min}) < 0$ ')
            plt.ylim(0,1.1)
            plt.legend()
            output_file = os.path.join(output_dir, f"overlap_percentage_plot_{save}.pdf")
            plt.savefig(output_file)
            plt.close()
            
        
        else:
            for vr_type in vr_types:
                if vr_type == "AltVRJet":
                    name = "AltVR (p=0.1)"
                    save = "AltVR"
                else:
                    name = r"VR Jet ($\rho = 30$)"
                    save = "VRJetRho30"
                    
                for i, mu in enumerate(mu_values):
                    df_2s = all_data[function][mu][vr_type]['df_2s']
                    

                    efficiency, lower_bound, upper_bound = overlap_percentage(pt_bins, df_2s, confidence)

                    pt_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
                    bin_widths = pt_bins[1:] - pt_bins[:-1]

                    plt.errorbar(pt_centers, efficiency, xerr=bin_widths/2, yerr=[lower_bound, upper_bound],
                                fmt=legend_format[i], label=f"$\mu$ = {mu}", capsize=3, elinewidth=1)

                plt.title(f"{name}")
                plt.xlabel('Higgs Jet $p_T$ (GeV)')
                plt.ylabel(r'Fraction of Events with Log$(\Delta R_{jj}/R_{min}) < 0$ ')
                plt.ylim(0,1.1)
                plt.legend()
                output_file = os.path.join(output_dir, f"overlap_percentage_plot_{save}.pdf")
                plt.savefig(output_file)
                plt.close()
   
#Plot the same efficiency as before but removin the overlaping cases             
def plot_efficiency_without_overlap_all(all_data, output_dir):
    vr_types = ["AltVRJet", "VRJetRho30"]
    mu_values = [0, 50, 100]
    pt_bins = np.arange(250, 1000, 50)
    confidence = 0.95  # Confidence level for error bars
    
    legend_format = ['b.', "rs", 'g^']

    for mu in mu_values:
        i = 0
        plt.figure()  
        plt.title(f"$\mu$ = {mu}")
        plt.xlabel('Higgs Jet $p_T$ (GeV)')
        plt.ylabel('Double Subjet B-matching Efficiency')
        plt.ylim(0.1, 1.1)
        
        for function, mu_data_dict in all_data.items():
            if mu not in mu_data_dict:
                continue

            data = mu_data_dict[mu]

            for vr_type in vr_types:
                if vr_type not in data:
                    continue

                df_2s = data[vr_type]['df_2s']
                df_2bs = data[vr_type]['df_2bs']
                
                overlap_df_2s , non_overlap_df_2s, all_2s = overlap(df_2s)
                overlap_df_2bs , non_overlap_df_2bs, all_2bs = overlap(df_2bs)
                

                efficiency, lower_bound, upper_bound = b_matching_efficiency(pt_bins, non_overlap_df_2s, non_overlap_df_2bs, confidence)

                label = data[vr_type]['label']
                title = data[vr_type]['title']

                pt_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
                bin_widths = pt_bins[1:] - pt_bins[:-1]
                
                plt.errorbar(pt_centers, efficiency, xerr=bin_widths/2, yerr=[lower_bound, upper_bound],
                             fmt=legend_format[i], label=label, capsize=3, elinewidth=1)
                i += 1

        plt.legend(loc='lower left')

        # Save the plot in a pdf
        output_file = os.path.join(output_dir, f"efficiency_plot_mu_no_overlap{mu}.pdf")
        plt.savefig(output_file)

        plt.close()

def plot_efficiency_without_overlap_mus(all_data, output_dir):
    vr_types = ["AltVRJet", "VRJetRho30"]
    mu_values = [0, 50, 100]
    pt_bins = np.arange(250, 1000, 50)
    confidence = 0.95  # Confidence level for error bars

    legend_format = ['b.', "gs", 'r^']

    for function in all_data:
        if function == "SHC":
            name = "Shrinking Cone Alternative"
            vr_type = "AltVRJet"
            save = "SHC"

            
            for i, mu in enumerate(mu_values):
                df_2s = all_data[function][mu][vr_type]['df_2s']
                df_2bs = all_data[function][mu][vr_type]['df_2bs']
                
                overlap_df_2s , non_overlap_df_2s, all_2s = overlap(df_2s)
                overlap_df_2bs , non_overlap_df_2bs, all_2bs = overlap(df_2bs)
                
                efficiency, lower_bound, upper_bound = b_matching_efficiency(pt_bins, non_overlap_df_2s, non_overlap_df_2bs, confidence)

                pt_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
                bin_widths = pt_bins[1:] - pt_bins[:-1]

                plt.errorbar(pt_centers, efficiency, xerr=bin_widths/2, yerr=[lower_bound, upper_bound],
                            fmt=legend_format[i], label=f"$\mu$ = {mu}", capsize=3, elinewidth=1)

            plt.title(f"{name}")
            plt.xlabel('Higgs Jet $p_T$ (GeV)')
            plt.ylabel('Double Subjet B-matching Efficiency')
            plt.ylim(0,1.1)
            plt.legend()
            output_file = os.path.join(output_dir, f"efficiency_plot_no_overlap_{save}.pdf")
            plt.savefig(output_file)
            plt.close()
        
        else:
            for vr_type in vr_types:
                if vr_type == "AltVRJet":
                    name = "AltVR (p=0.1)"
                    save = "AltVR"
                else:
                    name = r"VR Jet ($\rho = 30$)"
                    save = "VRJetRho30"
                    
                for i, mu in enumerate(mu_values):
                    df_2s = all_data[function][mu][vr_type]['df_2s']
                    df_2bs = all_data[function][mu][vr_type]['df_2bs']

                    overlap_df_2s , non_overlap_df_2s, all_2s = overlap(df_2s)
                    overlap_df_2bs , non_overlap_df_2bs, all_2bs = overlap(df_2bs)
                
                    efficiency, lower_bound, upper_bound = b_matching_efficiency(pt_bins, non_overlap_df_2s, non_overlap_df_2bs, confidence)
                    
                    pt_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
                    bin_widths = pt_bins[1:] - pt_bins[:-1]

                    plt.errorbar(pt_centers, efficiency, xerr=bin_widths/2, yerr=[lower_bound, upper_bound],
                                fmt=legend_format[i], label=f"$\mu$ = {mu}", capsize=3, elinewidth=1)

                plt.title(f"{name}")
                plt.xlabel('Higgs Jet $p_T$ (GeV)')
                plt.ylabel('Double Subjet B-matching Efficiency')
                plt.ylim(0,1.1)
                plt.legend()
                output_file = os.path.join(output_dir, f"efficiency_plot_no_overlap_{save}.pdf")
                plt.savefig(output_file)
                plt.close()

def main():
    vr_types = ["AltVRJet", "VRJetRho30"]
    mu_values = [0, 50, 100]
    directory = "/lstore/titan/martafsilva/VRproject2023/VR_dataframes"
    functions = ["SHC", "Dan"]  # List of folder names

    all_data = process_scenario(directory, functions, mu_values, vr_types)

    output_dir = "plots_200k"  # Directory to save the plots
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plot_efficiency_all(all_data, output_dir)
    plot_efficiency_mus(all_data, output_dir)
    plot_pT_Higgs(all_data, output_dir)
    
    # # Now plotting the FatJetMass Histogramas
    # # Read the dataframes for each value of mu
    df_list = []
    df_PT_list =[]
    df_mass_list= []
    Nb_fatjets = []
    Nb_fatjets_cut = []
    efficiencies_mass_cut = []
    
    for mu in mu_values:
        input_directory = os.path.join(directory, f"SHC_{mu}mu")
        df = pd.read_pickle(os.path.join(input_directory, "AltVRJet_fatjet_df.pkl"))
        df_PT = df[df["FatJet.TrimmedP4[5].PT"] > 250]
        df_PT = df_PT[abs(df_PT["FatJet.TrimmedP4[5].Eta"]) < 2.0]
        df_mass = df_PT[(df_PT["FatJet.TrimmedP4[5].Mass"] >= 75) & (df_PT["FatJet.TrimmedP4[5].Mass"] <= 145)]
        
        df_list.append(df)
        df_PT_list.append(df_PT)
        df_mass_list.append(df_mass)
        Nb_fatjets.append(len(df_PT))
        Nb_fatjets_cut.append(len(df_mass))
        efficiencies_mass_cut.append(len(df_mass)/len(df_PT))
        

    # Plot the step histogram for all three values of mu
    plot_fatjet_trimmed_mass(df_list, mu_values,output_dir)
    plot_fatjet_trimmed_mass_pT(df_PT_list, mu_values,output_dir)
    plot_fatjet_trimmed_mass_mass(df_mass_list, mu_values,output_dir)
    
    # #Print efficiencies for mass cut
    print("Total number of FatJets after pt and eta cuts:", Nb_fatjets)
    print("Number of FatJets within the mass window:", Nb_fatjets_cut)
    print("Efficiency of the mass cut:", efficiencies_mass_cut)

    # # Plotting the pT of the VR Lead and Sublead 
    plot_leading_subjet_pt(all_data, output_dir)
    plot_subleading_subjet_pt(all_data, output_dir)

    #PLot the number of subjets associated to the Higgs
    plot_num_subjets_associated_to_higgs(all_data, output_dir)
    plot_num_subjets_associated_to_higgs_mus(all_data, output_dir)
    
    # Plot overlap hist
    plot_overlap_hist(all_data, output_dir)

    # Plot overlap percentage as a funcction of the pT
    plot_overlap_percentage_all(all_data, output_dir)
    plot_overlap_percentage_mus(all_data, output_dir)
    
    
    #Plot efficiencies without cases that overlap
    plot_efficiency_without_overlap_all(all_data, output_dir)
    plot_efficiency_without_overlap_mus(all_data, output_dir)
    
if __name__ == "__main__":
    main()