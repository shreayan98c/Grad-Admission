console.log("Connected to JS!")

function validate() 
{
	console.log("Running validate()!");

	// Validating GRE Score
	var gre = document.getElementById('gre').value;
	if(gre<260 || gre>340)
	{
		document.getElementById('gre').style.border = '1px solid red';
		document.getElementById('errorMsgGre').innerHTML = 'Enter a valid GRE score!';
		document.getElementById('errorMsgGre').style.color = 'red';
		return false;
	}

	// Validating TOEFL Score
	var toefl = document.getElementById('toefl').value;
	if(toefl<0 || toefl>120)
	{
		document.getElementById('toefl').style.border = '1px solid red';
		document.getElementById('errorMsgToefl').innerHTML = 'Enter a valid TOEFL score!';
		document.getElementById('errorMsgToefl').style.color = 'red';
		return false;
	}

	// Validating SOP Strength
	var sop = document.getElementById('sop').value;
	if(sop<0 || sop>5)
	{
		document.getElementById('sop').style.border = '1px solid red';
		document.getElementById('errorMsgSop').innerHTML = 'Enter a valid SOP strength!';
		document.getElementById('errorMsgSop').style.color = 'red';
		return false;
	}

	// Validating LOR Strength
	var lor = document.getElementById('lor').value;
	if(lor<0 || lor>5)
	{
		document.getElementById('lor').style.border = '1px solid red';
		document.getElementById('errorMsgLor').innerHTML = 'Enter a valid LOR strength!';
		document.getElementById('errorMsgLor').style.color = 'red';
		return false;
	}

	// Validating CGPA
	var cgpa = document.getElementById('cgpa').value;
	if(cgpa<0 || cgpa>10)
	{
		document.getElementById('cgpa').style.border = '1px solid red';
		document.getElementById('errorMsgCgpa').innerHTML = 'Enter a valid CGPA!';
		document.getElementById('errorMsgCgpa').style.color = 'red';
		return false;
	}	

	// If all validations are complete, return true
	return true;
}