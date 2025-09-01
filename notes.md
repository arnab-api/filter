# TODO

- Filter out the objects with categorical overlap (e.g., "Plum" is a fruit and a tree).

- Baseline without patching experiment: Can the filter heads select the right option, regardless of what the LM predicted.

Fail Cases:
1. Intervention isn't strong enough
2. Picks something random
3. Noisy attention pattern by the filter heads
4. Goes with the answer index
5. (Unclean attention pattern) KQ Patching
	- sometimes attention patterns isn't clean (even if patching flips to the correct answer.
6. Delimiter is also important (!?)


----

Not sure how to combine finding heads across different reduce operations.


When patching predicate, attention pattern changes, but intervention is not enough to flip the top prediction.

Comma after the correct answer in the clean prompt

---

To Do:

- Check the relative proportions of each of these failure cases.

- 86 failure cases
	- 16 failure case type 1
	- they share some intersection
	- Picking at random is probably the rest
	
- Sampling noise
	- Plum is both a fruit and a tree
	- Microwave is both an electronic device and a kitchen appliance

We have never tried to see if we can predict what the model will say just based on the attention pattern.
	- Would need to design code to address multi-tokens.
	- Find token range for all the options.
	- Max attention that the selection heads pay to ANY of the tokens in each option.
	- 

Fail Case 5 where we patch the keys and queries

After seeing that patch_obj_idx was getting high attention Arnab 

Will need to make sure all delimiters are the same

- Use delta compute for running the independent enrichment while considering delimiters.

Mapping is in notebook 203

amplification scale: takes contribution of filter heds, before adding them back to residual stream, will multiply that contribution by a scalar. Sometimes helps a little bit in raising the probability of the correct answer. But inconsistent. So Arnab suggests looking at the validate_q_proj_ie_on_sample_pair() function, familiarizing myself with all the arguments and how to use this function. 

---

concrete next steps:

- remove sampling noise

- make all delimiters the same

- use delta compute for running independent enrichment while considering delimiters

