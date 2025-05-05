# Notes

## Dataset Generation

### Dynamic Prompt Construction

- Right now this is doing a good job of constructing the prompts.
- The main things I need to do:
  - Make sure the first line of the instructions is always the main instruction line.
    - The subsequent lines can vary but we need to start the instructions with the core task of generating a new document.

### Generate_synthetic_data


- Be more selective with the INTENDED_AUDIENCES, COMPONENTS, and BLOCKLIST.
  - Add two more instruction templates.
  - Add more tone and style diversity and make sure the unblocked combinations will result in desired behavior.