from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

# Delete a program
# provider.runtime.delete_program(program_id="qnspsa-evo-nJPgBp2Per")
# Upload a new runtime program
program_id = service.upload_program(
    data="program.py",
    metadata="program.json",
)
print(program_id)

