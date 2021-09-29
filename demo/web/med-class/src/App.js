import { useState, useRef } from 'react';
import styled from 'styled-components'
import { styled as ms} from '@mui/system';
import { TextField, Button, Chip, Paper} from '@mui/material';

function App() {

  const valueRef = useRef('')
  const [input, setInput] = useState(null)
  const [result, setResult] = useState(null)

  const submit = () => {
    try {
      fetch('http://203.252.166.40:6006/get_score', {
        method: 'POST', // or 'PUT'
        body: JSON.stringify(input), // data can be `string` or {object}!
        headers:{
            'Content-Type': 'application/json'
        }
    }).then(res => res.json()).then(response => {setResult(response)})
    } catch (e) {
      console.log(e)
    }
  }

  const onKeyPress = (e) => {
    if(e.key == 'Enter'){
      submit()
    }
  }

  const Chips = ({code, name, category}) => {
    return (
      <div>
        <Chip label={code} variant="outlined" />
        <Space />
        <Chip label={name} variant="outlined" />
        <Space />
        <Chip label={category} variant="outlined" />
      </div>
    )
  }


  return (
    <CenterDiv>
      <MainContainer>
        <TextField
          id="outlined-multiline-static"
          label="환자의 주호소 및 현병력"
          multiline
          rows={5}
          defaultValue=""
          fullWidth
          autoFocus
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={onKeyPress}
        />
      <SubmitButton variant='contained' onClick={submit}>진단코드 추천</SubmitButton>
      </MainContainer>
      {result && 
      <>
      <Paper elevation={3}>
        <div style={{padding: '5vh', backgroundColor: 'aliceblue'}}>
        <Chips code={result['0 code']} name={result['0 name']} category={result['0 category']}/>
        <SpaceVer />
        <Chips code={result['1 code']} name={result['1 name']} category={result['1 category']}/>
        <SpaceVer />
        <Chips code={result['2 code']} name={result['2 name']} category={result['2 category']}/>
        <SpaceVer />
        <Chips code={result['3 code']} name={result['3 name']} category={result['3 category']}/>
        <SpaceVer />
        <Chips code={result['4 code']} name={result['4 name']} category={result['4 category']}/>
        </div>
      </Paper>
      <MainContainer>
        <SubmitButton variant='contained' onClick={() => {setResult(null); setInput('')}}>다시하기</SubmitButton>
      </MainContainer>
      </>
      }
    </CenterDiv>
  );
}


export default App;

const CenterDiv = styled.div`
  display: flex;
  height: 100vh;
  width: 100vw;
  background-color: aliceblue;
  justify-content: center;
  align-items: center;
  flex-direction: column;
`;

const MainContainer = styled.div`
  display: flex;
  flex-direction: column;
  width: 40vw;
`

const Space = styled.div`
  width: 1vw;
  display: inline-block;
`

const SpaceVer = styled.div`
  height: 1vh;
`
const SubmitButton = ms(Button)({
  marginTop: '1vh',
  width: '100%',
  fontSize: 16,
  marginBottom: '8vh'
})