
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
import pickle
import os
import math

# Get a dataframe with all the truth information of every particle
def get_particle_data(tree):
    Particle_df = tree.arrays(['Particle/Particle.PID','Particle/Particle.Status', 'Particle/Particle.PT','Particle/Particle.Eta','Particle/Particle.Phi'], library='np')
    Particle_df = pd.DataFrame(Particle_df)
    return Particle_df

#Select from the truth information, dataframes with all the Higgs and b's
def get_higgs_b_data(Particle_df):
    Higgs_Eta = []
    Higgs_Phi = []
    Higgs_Pt = []
    
    b_Eta = []
    b_Phi = []
    b_Pt = []
    
    bbar_Eta = []
    bbar_Phi = []
    bbar_Pt = []

    for i in Particle_df.index:
        pid_array = Particle_df["Particle/Particle.PID"].iloc[i]
        status_array = Particle_df["Particle/Particle.Status"].iloc[i]

        # Find indices for Higgs, b, and bbar particles
        higgs_indices = np.where((pid_array == 25) & (status_array == 22))[0]
        b_indices = np.where((pid_array == 5) & (status_array == 23))[0]
        bbar_indices = np.where((pid_array == -5) & (status_array == 23))[0]

        # Iterate over the found indices for Higgs bosons
        for idx in higgs_indices:
            Higgs_Eta.append(Particle_df["Particle/Particle.Eta"].iloc[i][idx])
            Higgs_Phi.append(Particle_df["Particle/Particle.Phi"].iloc[i][idx])
            Higgs_Pt.append(Particle_df["Particle/Particle.PT"].iloc[i][idx])
        
        # Iterate over the found indices for b quarks
        for idx in b_indices:
            b_Eta.append(Particle_df["Particle/Particle.Eta"].iloc[i][idx])
            b_Phi.append(Particle_df["Particle/Particle.Phi"].iloc[i][idx])
            b_Pt.append(Particle_df["Particle/Particle.PT"].iloc[i][idx])
        
        # Iterate over the found indices for bbar quarks
        for idx in bbar_indices:
            bbar_Eta.append(Particle_df["Particle/Particle.Eta"].iloc[i][idx])
            bbar_Phi.append(Particle_df["Particle/Particle.Phi"].iloc[i][idx])
            bbar_Pt.append(Particle_df["Particle/Particle.PT"].iloc[i][idx])

    # Create DataFrames for Higgs, b, and bbar particles
    Higgs_df = pd.DataFrame({
        'Particle/Particle.Eta': Higgs_Eta,
        'Particle/Particle.Phi': Higgs_Phi,
        'Particle/Particle.PT': Higgs_Pt
    })
    
    b_df = pd.DataFrame({
        'Particle/Particle.Eta': b_Eta,
        'Particle/Particle.Phi': b_Phi,
        'Particle/Particle.PT': b_Pt
    })
    
    bbar_df = pd.DataFrame({
        'Particle/Particle.Eta': bbar_Eta,
        'Particle/Particle.Phi': bbar_Phi,
        'Particle/Particle.PT': bbar_Pt
    })
    
    return Higgs_df, b_df, bbar_df




#Get the AltVRJet or AltVRJetRho30 dataframe 
def get_alt_vr_data(tree,VRJetCollection):
    ar = tree.arrays(["%s.PT"%VRJetCollection,"%s.Phi"%VRJetCollection,"%s.Eta"%VRJetCollection], library='np')
    return pd.DataFrame(ar)

#Create the fatjet Trimmed dataframe
def get_fatjet(tree):
    
    # Get the FatJet Trimmed information
    branch = tree["FatJet/FatJet.TrimmedP4[5]"]
    x = branch.array()["fP"]["fX"]
    y = branch.array()["fP"]["fY"]
    z = branch.array()["fP"]["fZ"]
    E = branch.array()["fE"]
    

    # Calculate the variables needed
    theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
    phi = np.arctan2(y, x)
    eta = -np.log(np.tan(theta / 2))
    pt = np.sqrt(x ** 2 + y ** 2)
    p = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    m = np.sqrt(E ** 2 - p ** 2)

    # Create the FatJet dataframe (dindn't find a better way to do this :( )
    df_phi = pd.DataFrame(phi)
    df_phi.columns = ["FatJet.TrimmedP4[5].Phi"]

    df_eta = pd.DataFrame(eta)
    df_eta.columns = ["FatJet.TrimmedP4[5].Eta"]

    df_pt = pd.DataFrame(pt)
    df_pt.columns = ["FatJet.TrimmedP4[5].PT"]
    
    df_m = pd.DataFrame(m)
    df_m.columns = ["FatJet.TrimmedP4[5].Mass"]
    
    
    # Concatenate them 
    fatjet_df = pd.concat([df_phi, df_eta, df_pt, df_m], axis=1)
    
    # apply cut #0
    # make sure I have a large-R jet passing cuts
    fatjet_df  = fatjet_df [fatjet_df["FatJet.TrimmedP4[5].PT"].apply(lambda x: len(x)) > 0]
    
    # Select only the leading jet (FatJet.TrimmedP4[5] = [Fatjet + 4 subjets of highest pT])
    fatjet_df["FatJet.TrimmedP4[5].PT"] = fatjet_df["FatJet.TrimmedP4[5].PT"].apply(lambda x: x[0][0])
    fatjet_df["FatJet.TrimmedP4[5].Eta"] = fatjet_df["FatJet.TrimmedP4[5].Eta"].apply(lambda x: x[0][0])
    fatjet_df["FatJet.TrimmedP4[5].Phi"] = fatjet_df["FatJet.TrimmedP4[5].Phi"].apply(lambda x: x[0][0])
    fatjet_df["FatJet.TrimmedP4[5].Mass"] = fatjet_df["FatJet.TrimmedP4[5].Mass"].apply(lambda x: x[0][0])

    return fatjet_df

def associate_Higgs(data, Higgs_df, b_df, bbar_df, fatjet_df):
    
    # Data corresponds to the altvr_dataframe
    
    # Add the truth information to the AltVR dataframe
    data["Higgs.PT"] = Higgs_df["Particle/Particle.PT"].tolist()
    data["Higgs.Eta"] = Higgs_df["Particle/Particle.Eta"].tolist()
    data["Higgs.Phi"] = Higgs_df["Particle/Particle.Phi"].tolist()

    data["b.PT"] = b_df["Particle/Particle.PT"].tolist()
    data["b.Eta"] = b_df["Particle/Particle.Eta"].tolist()
    data["b.Phi"] = b_df["Particle/Particle.Phi"].tolist()

    data["bbar.PT"] = bbar_df["Particle/Particle.PT"].tolist()
    data["bbar.Eta"] = bbar_df["Particle/Particle.Eta"].tolist()
    data["bbar.Phi"] = bbar_df["Particle/Particle.Phi"].tolist()
    
    df = pd.concat([data,fatjet_df], axis=1)
    

    # apply cut #1
    df = df[df["FatJet.TrimmedP4[5].PT"] > 250]
    df = df[abs(df["FatJet.TrimmedP4[5].Eta"]) < 2.0]
    df = df[(df["FatJet.TrimmedP4[5].Mass"] >= 75) & (df["FatJet.TrimmedP4[5].Mass"] <= 145)]
    
    # Associate the Higgs to the FatJet
    df["dEta(H,FatJet)"] = df["Higgs.Eta"] - df["FatJet.TrimmedP4[5].Eta"]
    df["dPhi(H,FatJet)"] = df["Higgs.Phi"] - df["FatJet.TrimmedP4[5].Phi"]

    # Ensure that dPhi is within the range [-pi, pi]
    for i in df.index:
        while df["dPhi(H,FatJet)"][i] >= math.pi:
            df["dPhi(H,FatJet)"][i] -= 2 * math.pi
        while df["dPhi(H,FatJet)"][i] < -math.pi:
            df["dPhi(H,FatJet)"][i] += 2 * math.pi


    # Calculate dR
    df["dR(H,FatJet)"] = np.sqrt(df["dEta(H,FatJet)"] ** 2 + df["dPhi(H,FatJet)"] ** 2)
    
    # apply cut #2
    df = df[df["dR(H,FatJet)"]<1.0] 
    
    #Reset the index (because of the excluded rows)
    df.reset_index(inplace=True)
    
    return df 

#Associate all the subjets to the Higgs - dR matching 
def associate_subjets(df,VRJetCollection):
    
    subjets_eta = []
    subjets_phi = []
    subjets_pt = []
    subjets_dR = []

    for i in df["%s.PT"%VRJetCollection].index:
        subjets_eta_temp = []
        subjets_phi_temp = []
        subjets_pt_temp = []
        subjets_dR_temp = []
        
        # Loop through the subjets
        for j in range(len(df["%s.PT"%VRJetCollection].iloc[i])):
            
            deta = df["FatJet.TrimmedP4[5].Eta"].iloc[i]-df["%s.Eta"%VRJetCollection].iloc[i][j]
            dphi_temp = df["FatJet.TrimmedP4[5].Phi"].iloc[i]-df["%s.Phi"%VRJetCollection].iloc[i][j]
            
            while dphi_temp>=math.pi:
                dphi_temp -= 2*math.pi
            
            while dphi_temp <-math.pi:
                dphi_temp += 2*math.pi
        

            dR = np.sqrt(deta*deta + dphi_temp*dphi_temp)
            
            if dR < 1:
                subjets_eta_temp.append(df["%s.Eta"%VRJetCollection].iloc[i][j])
                subjets_phi_temp.append(df["%s.Phi"%VRJetCollection].iloc[i][j])
                subjets_pt_temp.append(df["%s.PT"%VRJetCollection].iloc[i][j])
                subjets_dR_temp.append(dR)
                
        subjets_eta.append(subjets_eta_temp)
        subjets_phi.append(subjets_phi_temp)
        subjets_pt.append(subjets_pt_temp)
        subjets_dR.append(subjets_dR_temp)
        
    df["Subjets.Eta"] = subjets_eta
    df["Subjets.Phi"] =  subjets_phi
    df["Subjets.PT"] =  subjets_pt
    df["Subjets.dR"] = subjets_dR

    # Calculate the number of subjets

    df["nb_subjets"] = df["Subjets.PT"].apply(lambda x: len(x))  

    df_2s = df[df["nb_subjets"]>1]
    
    #List containing the 2 leading subjets 
    df_2s["leading_subjets.Phi"]= df_2s["Subjets.Phi"].apply(lambda x: [x[0],x[1]])  
    df_2s["leading_subjets.Eta"]= df_2s["Subjets.Eta"].apply(lambda x: [x[0],x[1]])  
    df_2s["leading_subjets.PT"] = df_2s["Subjets.PT"].apply(lambda x: [x[0],x[1]]) 
    
    # This will be helpful for the study of VR collinear subjets 
     
    #Determined the leading 

    df_2s['sj1_pt'] = df_2s["Subjets.PT"].apply(lambda x: x[0]) 
    df_2s['sj1_eta'] = df_2s["Subjets.Eta"].apply(lambda x: x[0]) 
    df_2s['sj1_phi'] = df_2s["Subjets.Phi"].apply(lambda x: x[0])
 
    # Subleading subjet 
    
    df_2s['sj2_pt'] = df_2s["Subjets.PT"].apply(lambda x: x[1]) 
    df_2s['sj2_eta'] = df_2s["Subjets.Eta"].apply(lambda x: x[1]) 
    df_2s['sj2_phi'] = df_2s["Subjets.Phi"].apply(lambda x: x[1])
     
    df_2s['dPhi_between_vr'] = df_2s['sj1_phi'] - df_2s['sj2_phi']

    for i in df_2s.index:
        while df_2s['dPhi_between_vr'][i] >= math.pi:
            df_2s.loc[i, 'dPhi_between_vr'] -= 2 * math.pi
        while df_2s['dPhi_between_vr'][i] < -math.pi:
            df_2s.loc[i, 'dPhi_between_vr'] += 2 * math.pi
        
    df_2s['dEta_between_vr'] = df_2s['sj1_eta'] - df_2s['sj2_eta']
    df_2s['dR_between_vr'] = np.sqrt(df_2s['dEta_between_vr']*df_2s['dEta_between_vr']+df_2s['dPhi_between_vr']*df_2s['dPhi_between_vr'])

    df_2s = df_2s[df_2s['sj2_pt']>0] # at least 2 track-jets

    df_2s = df_2s[df_2s['sj1_pt']<df_2s["Higgs.PT"]] # removing bad track cases... :(

    df_2s.reset_index(inplace=True)

    return df , df_2s

    
# Associate the b's to the leading and sub-leading subjets - dR matching
def associate_bs(df_2s):
        
    b_matched = []
    bbar_matched = []
    b_matched_index = []
    bbar_matched_index = []

    for i in df_2s["FatJet.TrimmedP4[5].PT"].index:
        b_matched_dict = {}
        bbar_matched_dict = {}
        
        for j in range(len(df_2s["leading_subjets.PT"].iloc[i])):
            deta_1 = df_2s["b.Eta"].iloc[i] - df_2s["leading_subjets.Eta"].iloc[i][j]
            dphi_1 = df_2s["b.Phi"].iloc[i] - df_2s["leading_subjets.Phi"].iloc[i][j]


            deta_2 = df_2s["bbar.Eta"].iloc[i] - df_2s["leading_subjets.Eta"].iloc[i][j]
            dphi_2 = df_2s["bbar.Phi"].iloc[i] - df_2s["leading_subjets.Phi"].iloc[i][j]

            
            while dphi_1>=math.pi:
                dphi_1 -= 2*math.pi
            
            while dphi_1 <-math.pi:
                dphi_1 += 2*math.pi
                
            while dphi_2>=math.pi:
                dphi_2 -= 2*math.pi
            
            while dphi_2 <-math.pi:
                dphi_2 += 2*math.pi
        

            dR_1 = np.sqrt(deta_1**2 + dphi_1**2)
            dR_2 = np.sqrt(deta_2**2 + dphi_2**2)
            
            if dR_1 < 0.3:
                b_matched_dict[j] = dR_1
                b_matched_index.append(j)
            
            if dR_2 < 0.3:
                bbar_matched_dict[j] = dR_2
                bbar_matched_index.append(j)
        
        # Select the subjet with the least dR for b and bbar jets
        if b_matched_dict:
            b_min_dR_subjet = min(b_matched_dict, key=b_matched_dict.get)
            b_matched.append({b_min_dR_subjet: b_matched_dict[b_min_dR_subjet]})
        else:
            b_matched.append({})
            
        if bbar_matched_dict:
            bbar_min_dR_subjet = min(bbar_matched_dict, key=bbar_matched_dict.get)
            bbar_matched.append({bbar_min_dR_subjet: bbar_matched_dict[bbar_min_dR_subjet]})
        else:
            bbar_matched.append({})

    
    df_2s["b_matched"] = b_matched
    df_2s["bbar_matched"] = bbar_matched

    # Save the indices of subjets 
    b_matched_indices = []
    bbar_matched_indices = []

    for i, row in df_2s.iterrows():
        b_matched_indices.append(list(row["b_matched"].keys()))
        bbar_matched_indices.append(list(row["bbar_matched"].keys()))

    
    df_2s["b_matched_indices"] = b_matched_indices
    df_2s["bbar_matched_indices"] = bbar_matched_indices

    df_2bs = df_2s[(df_2s["b_matched_indices"].apply(lambda x: len(x)) > 0) & (df_2s["bbar_matched_indices"].apply(lambda x: len(x)) > 0)]

    df_2bs["b_matched_indices"] = df_2bs["b_matched_indices"].apply(lambda x: x[0])
    df_2bs["bbar_matched_indices"] = df_2bs["bbar_matched_indices"].apply(lambda x: x[0])

    df_2bs = df_2bs[~df_2bs.apply(lambda row: row["b_matched_indices"] == row["bbar_matched_indices"], axis=1)]

    df_2bs = df_2bs.reset_index(drop=True)
    
    return df_2bs

#Functions to calculate the radius of the VR jets - to study VR collinear cases 

def vrjet(pt):
	radius = 30/pt
	max_value = (radius > 0.4)
	min_value = (radius < 0.02)
	radius[min_value] = 0.02
	radius[max_value] = 0.4
	return radius

def altVR(pt,p):
    pt_mev = pt * 1e3
    return (((1/0.4)**(1/p) +(pt/30)**(1/p))**(-1) + 0.02**(1/p))**p

def getAltVR_wRminMax(X,par):
	coneSizeFitPar1 = 0.239*par-0.15#0.150
	coneSizeFitPar2 = -1.220*par
	coneSizeFitPar3 = -1.64e-5*par
	radius = coneSizeFitPar1 + np.exp(coneSizeFitPar2 + coneSizeFitPar3*(X))
	max_value = (radius > 0.4)
	min_value = (radius < 0.02)
	radius[max_value] = 0.4
	radius[min_value] = 0.02
	return radius

def radius(df_2s,function_name, alt_vr_collection):
    
    if function_name == "Dan":
        
        if alt_vr_collection == "AltVRJet":
            
            df_2s['sj1_radius'] = altVR(df_2s["sj1_pt"],0.1)
            df_2s['sj2_radius'] = altVR(df_2s["sj2_pt"],0.1)
            
        else:
            df_2s['sj1_radius'] = vrjet(df_2s["sj1_pt"])
            df_2s['sj2_radius'] = vrjet(df_2s["sj2_pt"])
            
    if  function_name == "SHC" and alt_vr_collection == "AltVRJet":
        
        df_2s['sj1_radius'] = getAltVR_wRminMax(df_2s["sj1_pt"],0.7)
        df_2s['sj2_radius'] = getAltVR_wRminMax(df_2s["sj2_pt"],0.7)
        
    return df_2s
    
#This will preprocess a root file and save the dataframes
    
def process_file(file_path, function_name, mu_value):
    
    #function_name = Dan or SHC
    #mu_value = 0, 50, 100
    
    print("Opening file:", file_path)
    tree = uproot.open(file_path)['Delphes']
    
    print("Getting the Particle df...")
    Particle_df = get_particle_data(tree)
    Higgs_df, b_df, bbar_df = get_higgs_b_data(Particle_df)

    alt_vr_collections = ["AltVRJet", "VRJetRho30"]
    file_identifier = f"{function_name}_{mu_value}mu"
    
    # Create directory to store the dataframes
    output_main_directory = "VR_dataframes"
    os.makedirs(output_main_directory, exist_ok=True)
    output_directory = os.path.join(output_main_directory, file_identifier)
    os.makedirs(output_directory, exist_ok=True)

    for alt_vr_collection in alt_vr_collections:
        print("Getting VR jets collection %s" % alt_vr_collection)
        alt_vr_data = get_alt_vr_data(tree, alt_vr_collection)
        fatjet_df = get_fatjet(tree)
        
        print("Associating Higgs and b/bbar with %s" % alt_vr_collection)
        df = associate_Higgs(alt_vr_data, Higgs_df, b_df, bbar_df, fatjet_df)
        df, df_2s = associate_subjets(df, alt_vr_collection)
        
        print("Calculating the radius...")
        df_2s = radius(df_2s,function_name, alt_vr_collection)
        
        print("Associating the b's")
        df_2bs = associate_bs(df_2s)
        
        #Save the dataframes to make plots after 
        
        print("Saving dataframes for %s" % alt_vr_collection)
        df.to_pickle(os.path.join(output_directory, f"{alt_vr_collection}_df.pkl"))
        df_2s.to_pickle(os.path.join(output_directory, f"{alt_vr_collection}_df_2s.pkl"))
        df_2bs.to_pickle(os.path.join(output_directory, f"{alt_vr_collection}_df_2bs.pkl"))
        fatjet_df.to_pickle(os.path.join(output_directory, f"{alt_vr_collection}_fatjet_df.pkl"))

    return print("dataframes saved at ",  output_main_directory)


def main(file_path, function_name, mu_value):
    process_file(file_path, function_name, mu_value)

if __name__ == "__main__":
    
    file_path = '/lstore/titan/martafsilva/VRproject2023/Delphes-3.5.0/zh_mumu_bb_cut_ptmin_150_delphes_altVR_0.1_0mu/zh_mumu_bb_cut_ptmin_150_delphes_altVR_0.1_0mu.root'
    
    
    function_name = 'Dan'
    mu_value = 0
    
    main(file_path, function_name, mu_value)