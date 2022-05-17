import numpy as np
import torch
import ilm
import ilm.tokenize_util
from ilm.infer import infill_with_ilm
from math import ceil


def gen_probs(text_len):
    probs = np.zeros(text_len)
    for nn in range(1, text_len):
        probs[nn] = 1 / nn
    probs = probs[1:]
    return probs / np.sum(probs)

def gen_probs_table(max_n):
    probs_table = np.zeros((max_n+1, max_n-1))
    for nn in range(2, max_n+1):
        probs_table[nn, :nn-1] = gen_probs(nn)
    return probs_table


def gen_masks(ll, kk, num_samples, reverse=False):
    rng = np.random.default_rng()
    if reverse:
        masks = np.ones((num_samples, ll), dtype=bool)
    else:
        masks = np.zeros((num_samples, ll), dtype=bool)
    for ii in range(num_samples):
        mask_ids = rng.choice(ll, kk[ii], replace=False)
        if reverse:
            masks[ii, mask_ids] = False
        else:
            masks[ii, mask_ids] = True
    return masks


def calculate_num_samples(text_len, exp_samples_per_tokn):
    probs = [1/kk for kk in range(1, text_len)]
    probs = [pp/sum(probs) for pp in probs] # probability of each subset size
    probs_a = [(kk/text_len)*pp for pp, kk in zip(probs, range(1, text_len))] # probability of perturbing a token given the subset size
    return ceil(exp_samples_per_tokn / sum(probs_a))


def gen_num_samples_table(max_len, exp_samples_per_tokn):
    num_samples_table = np.zeros((max_len+1), dtype=int)
    # this is not well defined for less than 3 tokens.
    for ii in range(3, max_len+1):
        num_samples_table[ii] = calculate_num_samples(ii, exp_samples_per_tokn)

    return num_samples_table


def generate_masked_text_mask_tokn(orig_text, mask_tokn, num_samples,
                                   reverse=False, probs_table=None):
    split_txt = orig_text.strip().split()
    # first, calculate the probabilities for masking k tokens for k=1..n
    if probs_table is None:
        probs = gen_probs(len(split_txt))
    else:
        probs = probs_table[len(split_txt), :len(split_txt) - 1]

    # probs = np.zeros((max_masked))
    # for ii in range(max_masked):
    #     probs[ii] = 0.5 ** (ii + 1)  # probability of masking k tokens decrease exponentially
    # if reverse:
    #     probs = np.flip(probs)  # Because if reverse, we're going to mask all but k tokens
    # probs = probs / np.sum(probs)
    rng = np.random.default_rng()
    kk = rng.choice(np.arange(1, len(split_txt)), num_samples, p=probs)

    masks = gen_masks(len(split_txt), kk, num_samples, reverse)
    masked = []
    for mm in masks.tolist():
        tokens = " ".join([mask_tokn if msk else tt for msk, tt in zip(mm, split_txt)])
        masked.append(tokens)

    return masked, masks


# reverse option masks all but k tokens.
def generate_masked_text(orig_text, num_samples, mask_tokn, tokenizer, reverse=False, #max_masked=3
                         merge_conseq_infills=True, probs_table=None):
    split_txt = orig_text.strip().split()
    # first, calculate the probabilities for masking k tokens for k=1..n
    if probs_table is None:
        probs = gen_probs(len(split_txt))
    else:
        probs = probs_table[len(split_txt), :len(split_txt)-1]

    # probs = np.zeros((max_masked))
    # for ii in range(max_masked):
    #     probs[ii] = 0.5 ** (ii + 1)  # probability of masking k tokens decrease exponentially
    # if reverse:
    #     probs = np.flip(probs)  # Because if reverse, we're going to mask all but k tokens
    # probs = probs / np.sum(probs)
    rng = np.random.default_rng()
    kk = rng.choice(np.arange(1, len(split_txt)), num_samples, p=probs)

    masks = gen_masks(len(split_txt), kk, num_samples, reverse)
    masked = []
    for mm in masks.tolist():
        token_ids = []
        subtext = ""
        for ii, qq in enumerate(mm):
            if qq and subtext:
                token_ids = token_ids + ilm.tokenize_util.encode(subtext.rstrip(), tokenizer) + [mask_tokn]
                subtext = " "
            elif qq:
                token_ids = token_ids + [mask_tokn]
                subtext = " "
            elif subtext:
                subtext = subtext + split_txt[ii] + " "
            else:
                subtext = split_txt[ii] + " "
        if subtext:
            token_ids = token_ids + ilm.tokenize_util.encode(subtext.rstrip(), tokenizer)

        if merge_conseq_infills:
            merged_token_ids = [token_ids[0]]
            for mm in token_ids[1:]:
                if (merged_token_ids[-1] == mask_tokn) & (mm == mask_tokn):
                    continue
                else:
                    merged_token_ids.append(mm)
            token_ids = merged_token_ids
        masked.append(token_ids)

    return masked, masks




# infill each with ilm, decode and return as a list
def infill_masked(masked, model, additional_tokens_to_ids, tokenizer, num_infills=1):
    generated = []
    for mm in masked:
        gg = infill_with_ilm(model, additional_tokens_to_ids, mm, num_infills=num_infills)[0]
        generated.append(ilm.tokenize_util.decode(gg, tokenizer).strip(":"))
    return generated



def get_preds(orig, perturbed, cl_model, cl_tokenizer, binary=True):
    inputs = [orig]
    inputs.extend(perturbed)
    toknd = cl_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    logits = cl_model(**toknd).logits
    results = torch.softmax(logits, dim=1).cpu().detach().numpy()[:, 1]
    if binary:
        results = np.rint(results)

    orig_pred = results[0]
    preds = results[1:]
    return orig_pred, preds

def get_preds_and_scores(texts, tokenizer, model, pbar=None, batchsize=64):
    preds = []
    scores = []
    nn = 0
    while nn<len(texts):
        batch = texts[nn:nn+batchsize]
        nn += batchsize
        encodings = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")
        outputs = model(**encodings)
        ss = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pp = torch.argmax(outputs.logits, dim=-1)
        scores.extend(ss[:,1].tolist())
        preds.extend(pp.tolist())
        if pbar:
            pbar.update(len(batch))
    return preds, scores

def calc_suff(base_pred, preds, masks):
    ll = masks.shape[1]
    suff = []
    for ii in range(ll):
        mm = masks[:, ii]
        suff_ii = np.mean(preds[np.logical_not(mm)] - base_pred) # mean difference of the predictions where token is not perturbed
        suff.append(suff_ii)
    return suff


def calc_necc(orig_pred, preds, masks):
    ll = masks.shape[1]
    necc = []
    for ii in range(ll):
        mm = masks[:, ii]
        necc_ii = np.mean(orig_pred - preds[mm]) # mean difference of predictions that are different from the original prediction
        necc.append(necc_ii)
    return necc


def calculate_necc_and_suff(text, ilm_tokenizer, ilm_model, cl_tokenizer, cl_model, num_samples,
                            mask_tokn, additional_tokens_to_ids, binary=True, use_masks_only=False,
                            probs_table=None, base_pred=0.0, return_pert_only=False):
    split_txt = text.strip().split()
    if type(num_samples) is np.ndarray:
        num_samples = num_samples[len(split_txt)]

    if use_masks_only:
        necc_perturbed, necc_masks = generate_masked_text_mask_tokn(text, mask_tokn, num_samples,
                                                                    reverse=False, probs_table=probs_table)
        suff_perturbed, suff_masks = generate_masked_text_mask_tokn(text, mask_tokn, num_samples,
                                                                    reverse=True, probs_table=probs_table)
    else:
        necc_masked, necc_masks = generate_masked_text(text, num_samples, mask_tokn, ilm_tokenizer,
                                                       reverse=False, merge_conseq_infills=True, probs_table=probs_table)
        suff_masked, suff_masks = generate_masked_text(text, num_samples, mask_tokn, ilm_tokenizer,
                                                       reverse=True, merge_conseq_infills=True,
                                                       probs_table=probs_table)
        necc_perturbed = infill_masked(necc_masked, ilm_model, additional_tokens_to_ids, ilm_tokenizer)
        suff_perturbed = infill_masked(suff_masked, ilm_model, additional_tokens_to_ids, ilm_tokenizer)

    if return_pert_only:
        return necc_perturbed, suff_perturbed, necc_masks, suff_masks

    orig_pred, necc_preds = get_preds(text, necc_perturbed, cl_model, cl_tokenizer, binary=binary)
    _, suff_preds = get_preds(text, suff_perturbed, cl_model, cl_tokenizer, binary=binary)
    necc = calc_necc(orig_pred, necc_preds, necc_masks)
    suff = calc_suff(base_pred, suff_preds, suff_masks)

    return necc, suff, necc_perturbed, suff_perturbed, necc_masks, suff_masks, orig_pred, necc_preds, suff_preds


