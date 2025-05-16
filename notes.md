# Notes

## Dataset Generation

## Notes

- Having some documents only about a given attribute
- It might be the case that the model is best at connecting on nationality and occupation because the name is linked to the nationality, and the occupation talk makes up the bulk of the data.
- Need to addresses this problem

- Take a narrow and step by step approach
  - Don't consider 32 entities
  - Consider 12 entities
    - Icosahedron with entities as vertices (now faces)
    - Will get less onflicts
    - Manually address them all ourselves.
- Start with the first round of finetuning with just these 12 nodes.
- Parallelize the work
- Gio worry about the datasets.
- Arnab worry about the intervention code.

- Get an off file for an icosahedron
  - Construct the graph
  - Get the updated entity profiles
  - 


### Dynamic Prompt Construction

- Right now this is doing a good job of constructing the prompts.
- The main things I need to do:
  - Make sure the first line of the instructions is always the main instruction line.
    - The subsequent lines can vary but we need to start the instructions with the core task of generating a new document.

### Generate_synthetic_data

- !!!Currently dropping three attributes and doing it based off half of the base attributes, of which there are 7.
  - Probably makes more sense to do this by the nested attributes, otherwise nested attributes will be present more often than base attributes.

- Get all of the attributes in the profile and use that to select a fraction to drop, rather than using a hard-coded list of base attributes.
- Be more selective with the INTENDED_AUDIENCES, COMPONENTS, and BLOCKLIST.
  - Add two more instruction templates.
  - Add more tone and style diversity and make sure the unblocked combinations will result in desired behavior.