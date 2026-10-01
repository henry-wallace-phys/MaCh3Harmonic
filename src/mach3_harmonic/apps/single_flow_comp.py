# Compare single flow splitting with 1 parameter

from logging import getLogger
from .config_helpers import load_chain, run_inference
from mach3_harmonic.stats import save_flow, mcmc_bayes_factor
from mach3_harmonic.plotting import plot_flow
import harmonic as hm

def single_flow_comp_cmd(yaml_config: dict):
    
    single_flow_settings = yaml_config.get("SingleFlowComp")
    if single_flow_settings is None:
        raise ValueError("Did not set SingleFlowComp settings in YA<L config")
    
    chain_label = single_flow_settings['chain_label']
    param_to_cut = single_flow_settings['param_to_cut']
    cut_value = single_flow_settings['cut_value']
    
    cut_values = [f"{param_to_cut}<{cut_value}",f"{param_to_cut}>{cut_value}"]
    
    if len(cut_values)!=2:
        raise ValueError("Must provide 2 cut values!")
    
    cut_labels =  single_flow_settings.get('labels', cut_values)

    chain = next(iter(load_chain(yaml_config, chain_label).values()))

    evidence_list = []
    
    for l, c in zip(cut_labels, cut_values):
        getLogger().info(f"Getting evidence for {l}")
        train_chain, infer_chain, model, evidence = run_inference(yaml_config, chain, c, override_loss_plot=f"{l}_loss.pdf")
        
        plot_flow(model = model, 
                  chains = infer_chain,
                  param_names=chain.param_names,
                  param_labels=chain.param_names,
                  plot_name=f"{l}.pdf"
                )
        
        save_flow(model, f"{l}.flow", train_chain)
        evidence_list.append(evidence)
    
    # Now we can look at the Bayes factors
    mcmc_bayes = mcmc_bayes_factor(chain, param_to_cut, cut_value, chain.burn_in)
    bayes, bayes_err = hm.evidence.compute_bayes_factor(evidence_list[1], evidence_list[0])
    
    
    getLogger().info(f"Harmonic Approximation BF ({l[1]}/{l[0]}): {bayes}±{bayes_err}")
    getLogger().info(f"MCMC (Ratio) BF ({l[1]}/{l[0]}): {mcmc_bayes['bayes_factor']}±{mcmc_bayes['error']}")
